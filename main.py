from fastapi import FastAPI, Request, Form, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch.nn.functional as F
import traceback
import os, base64, cv2
import numpy as np
import torch
import asyncio
from datetime import datetime
from io import BytesIO
from PIL import Image
from src.logic import (
    detect_faces_retinaface, extract_embedding, recognize_user,
    anti_spoof_check, mask_model, preprocess_input, img_to_array, blink_model,
    face_mesh, extract_eye_tensor, predict_avg_eye_status, is_face_straight_pnp,
    LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
)
from src.database import (
    insert_user, get_user_embeddings, insert_log_entry, fetch_users, fetch_punch_logs, fetch_all_users, delete_user_by_id, fetch_latest_log
)
from scipy.spatial.distance import cosine
from collections import deque
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
partial_registrations = {}

async def safe_send(websocket, message: str):
    if websocket.client_state.name != "CONNECTED":
        print(f"[WS SEND SKIPPED] Client already disconnected. Cannot send: '{message}'")
        return
    try:
        await websocket.send_text(message)
    except Exception as e:
        print(f"[WS SEND ERROR] Could not send '{message}' – {e}")


def decode_base64_image(data_url):
    header, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)
    img_array = np.array(Image.open(BytesIO(img_data)).convert("RGB"))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.get("/blink", response_class=HTMLResponse)
async def blink_gate(request: Request, target: str = Query(..., pattern="^(register|login|logout)$")):
    """
    Renders the blink-verification page.
    When the user blinks successfully, the page JS will redirect to /{target}.
    """
    return templates.TemplateResponse("blink.html",
                                      {"request": request,
                                       "target": target})

@app.websocket("/ws/blink-check")
async def blink_check_ws(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened for blink-check.")

    eye_pred_buffer = deque(maxlen=7)
    last_state = 1
    blink_counter = 0
    start_time = time.time()
    timeout_sec = 10
    alignment_status = None

    try:
        while True:
            if time.time() - start_time > timeout_sec:
                await safe_send(websocket, "timeout")
                break

            try:
                data_url = await websocket.receive_text()
            except WebSocketDisconnect:
                print("WebSocket client disconnected early.")
                break

            frame = decode_base64_image(data_url)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                print("[WARN] No face landmarks detected.")
                await safe_send(websocket, "face_not_straight")
                alignment_status = "not_straight"
                blink_counter = 0
                eye_pred_buffer.clear()
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            ih, iw = frame.shape[:2]
            is_straight = is_face_straight_pnp(landmarks, iw, ih)
            print(f"[STRAIGHT] is_straight={is_straight}")

            if is_straight:
                if alignment_status != "straight":
                    print("[✅] Face is now straight.")
                    await safe_send(websocket, "face_straight")
                    alignment_status = "straight"
            else:
                if alignment_status != "not_straight":
                    print("[↩️] Face not straight.")
                    await safe_send(websocket, "face_not_straight")
                    alignment_status = "not_straight"
                blink_counter = 0
                eye_pred_buffer.clear()
                continue

            # Extract eyes
            left_tensor, _ = extract_eye_tensor(frame, landmarks, LEFT_EYE_LANDMARKS)
            right_tensor, _ = extract_eye_tensor(frame, landmarks, RIGHT_EYE_LANDMARKS)

            if left_tensor is None or right_tensor is None:
                print("[WARN] Eye crop failed.")
                continue

            with torch.no_grad():
                input_tensor = torch.stack([left_tensor, right_tensor]).to("cpu")
                output = blink_model(input_tensor)
                prob = F.softmax(output, dim=1).cpu().numpy()
                avg_prob = np.mean(prob, axis=0)

                print(f"[BLINK] pred={np.argmax(avg_prob)}, open_conf={avg_prob[1]:.3f}, closed_conf={avg_prob[0]:.3f}")

                if avg_prob[0] > 0.7:
                    pred = 0  # Closed
                elif avg_prob[1] > 0.7:
                    pred = 1  # Open
                else:
                    pred = last_state

                eye_pred_buffer.append(pred)
                smoothed_pred = round(np.mean(eye_pred_buffer))
                print(f"pred={pred}, last_state={last_state}, blink_counter={blink_counter}")

                # Use raw pred (0 = closed, 1 = open) for transition check
                if pred == 0 and last_state == 1:
                    blink_counter += 1
                    print(f"[👁] Blink detected ({blink_counter}/1)")
                    if blink_counter >= 1:
                        print("[✅] Blink confirmed and sent.")
                        await safe_send(websocket, "blink_detected")
                        break
                else:
                    blink_counter = 0

                last_state = pred  # use raw prediction for tracking
                await asyncio.sleep(0.5)

    except Exception as e:
        print("Unhandled error during blink check:")
        import traceback
        traceback.print_exc()
        await safe_send(websocket, "error: " + str(e))
    finally:
        if not websocket.client_state.name == "DISCONNECTED":
            await websocket.close()
        print("WebSocket connection closed")


@app.get("/punch-log", response_class=HTMLResponse)
async def view_combined_log(request: Request):
    print("✅ /punch-log route hit")
    logs = fetch_punch_logs()
    users = fetch_all_users()
    print("👤 USERS from DB:", users)
    return templates.TemplateResponse("punch_log.html", {"request": request, "logs": logs, "users": users})


@app.get("/punch-log/edit/{log_id}", response_class=HTMLResponse)
async def edit_log(request: Request, log_id: int):
    from src.database import get_log_by_id
    log = get_log_by_id(log_id)
    if not log:
        raise HTTPException(status_code=404, detail="Log entry not found")
    return templates.TemplateResponse("edit_log.html", {"request": request, "log": log})


@app.post("/punch-log/update/{log_id}", response_class=HTMLResponse)
async def update_log(log_id: int, name: str = Form(...), spoof_status: str = Form(...), mask_status: str = Form(...), type: str = Form(...)):
    from src.database import update_log_entry
    update_log_entry(log_id, name, spoof_status, mask_status, type)
    return RedirectResponse(url="/punch-log", status_code=303)


@app.post("/users/delete/{user_id}")
async def delete_user(user_id: int):
    delete_user_by_id(user_id)
    return RedirectResponse(url="/punch-log", status_code=303)


@app.get("/{mode}", response_class=HTMLResponse)
async def show_camera(request: Request, mode: str):
    if mode not in ['register', 'login', 'logout']:
        raise HTTPException(status_code=404, detail="Invalid page")
    return templates.TemplateResponse("webcam.html", {"request": request, "mode": mode})


@app.post("/register", response_class=HTMLResponse)
async def register_user(
    request: Request,
    name: str = Form(...),
    image: str = Form(...),
    mode: str = Form(...)
):  
    # Validate name format
    if not name or len(name.strip()) < 2 or not name.replace(" ", "").isalpha():
        return "Invalid name. Only letters and spaces allowed."
    
    # ❌ Prevent duplicate name (fully registered or partially registered same mode)
    if get_user_embeddings(name):
        return f"Username '{name}' is already registered."
    
    if name in partial_registrations and mode in partial_registrations[name]:
        return f"{mode.title()} face already captured for '{name}'."
    
    # Validate mode
    if mode not in ["mask", "no_mask"]:
        raise HTTPException(status_code=400, detail="Invalid registration mode")
    
    # Decode and detect face
    frame = decode_base64_image(image)
    print("[DEBUG] Received image, decoding now.")
    faces = detect_faces_retinaface(frame)
    print(f"[DEBUG] Faces detected: {faces}")
    if not faces:
        return "No face detected."

    (x1, y1, x2, y2) = faces[0]
    face_crop = frame[y1:y2, x1:x2]
    label, _ = anti_spoof_check(frame)
    if label != 1:
        return "Spoof detected."

    # Mask prediction
    resized = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (224, 224))
    arr = preprocess_input(img_to_array(resized)[None, ...])
    mask_prob, no_mask_prob = mask_model.predict(arr)[0]
    is_masked = mask_prob > no_mask_prob
    detected_mode = "mask" if is_masked else "no_mask"

    if detected_mode != mode:
        return f"Incorrect face type. Expected: {mode}. Detected: {detected_mode}."

    # Get face embedding
    current_embedding = extract_embedding(face_crop)

    # ❌ Check duplicate face (must come BEFORE name check)
    for user in fetch_users():
        for stored_emb in [user["mask"], user["no_mask"]]:
            if stored_emb is not None:
                stored_emb = np.array(stored_emb, dtype=np.float32)
                dist = cosine(current_embedding, stored_emb)
                if dist < 0.3:
                    return f"This face is already registered under '{user['name']}'."

    # ✅ Save to partial memory
    if name not in partial_registrations:
        partial_registrations[name] = {}

    partial_registrations[name][mode] = current_embedding

    # ✅ Commit to DB if both modes are done
    if "mask" in partial_registrations[name] and "no_mask" in partial_registrations[name]:
        insert_user(
            name,
            mask_emb=partial_registrations[name]["mask"],
            no_mask_emb=partial_registrations[name]["no_mask"]
        )
        del partial_registrations[name]
        return f"Full registration completed for '{name}'."

    return f"{mode.title()} face captured. Now please {'wear' if mode == 'no_mask' else 'remove'} your mask and capture the other one."


@app.post("/auth")
async def authenticate_user(request: Request, image: str = Form(...), type: str = "login"):
    frame = decode_base64_image(image)
    faces = detect_faces_retinaface(frame)
    if not faces:
        return JSONResponse(content={"status": "error", "message": "No face detected."})

    (x1, y1, x2, y2) = faces[0]
    face_crop = frame[y1:y2, x1:x2]

    label, _ = anti_spoof_check(frame)
    if label != 1:
        return JSONResponse(content={"status": "error", "message": "Spoof detected."})

    resized = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (224, 224))
    arr = preprocess_input(img_to_array(resized)[None, ...])
    mask_prob, no_mask_prob = mask_model.predict(arr)[0]
    is_masked = mask_prob > no_mask_prob
    mask_label = "Mask" if is_masked else "No Mask"
    current_embedding = extract_embedding(face_crop)
    db_users = fetch_users()
    identity = recognize_user(current_embedding, is_masked, db_users)

    print("[🔍] Comparing against registered users:")
    for user in db_users:
        name = user["name"]
        stored_embedding = np.array(user["mask" if is_masked else "no_mask"], dtype=np.float32)
        score = 1 - cosine(current_embedding, stored_embedding)
        print(f" → {name}: similarity = {score:.4f}")

    if identity == "Unknown":
        return JSONResponse(content={
            "status": "unrecognized",
            "message": "User not recognized. Would you like to register?",
            "redirect": "/blink?target=register"
        })

    latest_log_type = fetch_latest_log(identity)
    if latest_log_type == type:
        return JSONResponse(content={
            "status": "error",
            "message": f"{identity} already {'logged in' if type == 'login' else 'logged out'}."
        })

    timestamp = datetime.now()
    insert_log_entry(identity, timestamp, "Real", mask_label, punch_type=type)

    return JSONResponse(content={
        "status": "ok",
        "message": f"{'Welcome' if type == 'login' else 'Goodbye'} {identity}."
    })