import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import re
import os
import csv
import mediapipe as mp
import torchvision.transforms as transforms
from retinaface import RetinaFace
from datetime import datetime
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.model import Model as BlinkModel
from collections import deque


mask_model = load_model("/Users/calebyeo/Downloads/4. Degree 3/Project/System/models/mask_detection_model/mask_detector.model.h5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recognition_model = torch.load("/Users/calebyeo/Downloads/4. Degree 3/Project/System/models/mask_recognition_model/InceptionResNetV1_ArcFace.pt", map_location=device, weights_only=False)
recognition_model.eval().to(device)
model_test = AntiSpoofPredict(0)
cropper = CropImage()
anti_spoof_model_dir = "/Users/calebyeo/Downloads/4. Degree 3/Project/System/models/anti_spoof_model"
RECOGNITION_THRESHOLD = 0.8

blink_model = BlinkModel(num_classes=2)
checkpoint = torch.load("/Users/calebyeo/Downloads/4. Degree 3/Project/System/models/blink_detection_model/model_11_96_0.1256.t7", map_location=torch.device("cpu"))
blink_model.load_state_dict(checkpoint['net'])
blink_model.eval().to("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
EYE_IMG_SIZE = 24
eye_pred_buffer = deque(maxlen=5)
last_state = 1

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]


# Structure: {"Alice": {"mask": embedding, "no_mask": embedding}}
if os.path.exists("user_embeddings.npy"):
    user_embeddings = np.load("user_embeddings.npy", allow_pickle=True).item()
else:
    user_embeddings = {}

log_file = "logs/punch_log.csv"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Name", "Timestamp", "Spoof_Status", "Mask_Status"])


def detect_faces_retinaface(frame):
    detections = RetinaFace.detect_faces(frame)
    faces = [det["facial_area"] for det in detections.values()] if isinstance(detections, dict) else []
    if not faces:
        print("[WARN] No face detected in frame.")
    return faces


def extract_embedding(face_img):
    face = cv2.resize(face_img, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face - 127.5) / 128.0
    face = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = recognition_model(face)
        embedding = output['embeddings'] if isinstance(output, dict) else output
        embedding = F.normalize(embedding).cpu().numpy()[0]
    return embedding


def recognize_user(current_embedding, is_masked, threshold=RECOGNITION_THRESHOLD):
    key = "mask" if is_masked else "no_mask"
    max_sim, identity = -1, "Unknown"
    for name, profiles in user_embeddings.items():
        if key not in profiles:
            continue
        db_emb = profiles[key]
        sim = np.dot(current_embedding, db_emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(db_emb))
        if sim > max_sim:
            max_sim, identity = sim, name
    print(f"[🔍] Highest cosine similarity ({key}): {max_sim:.4f}")
    return identity if max_sim >= threshold else "Unknown"


def anti_spoof_check(frame):
    detections = RetinaFace.detect_faces(frame)
    if not isinstance(detections, dict) or len(detections) == 0:
        print("[!] No face detected for anti-spoofing.")
        return 0, 0.0

    facial_area = list(detections.values())[0]["facial_area"]  # (x1, y1, x2, y2)
    x1, y1, x2, y2 = facial_area
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    target_h = y2 - y1
    target_w = int(target_h * 3 / 4)

    new_x1 = max(0, cx - target_w // 2)
    new_y1 = max(0, cy - target_h // 2)
    new_x2 = min(frame.shape[1], new_x1 + target_w)
    new_y2 = min(frame.shape[0], new_y1 + target_h)

    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h

    cropped_face = frame[new_y1:new_y2, new_x1:new_x2]
    cv2.imshow("AntiSpoof Input", cropped_face)

    bbox = model_test.get_bbox(cropped_face)
    prediction = np.zeros((1, 3))
    test_speed = 0

    for model_name in os.listdir(anti_spoof_model_dir):
        if not model_name.endswith('.pth'):
            continue
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": cropped_face,
            "bbox": bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True if scale else False,
        }

        img = cropper.crop(**param)

        start = time.time()
        pred = model_test.predict(img, os.path.join(anti_spoof_model_dir, model_name))
        prediction += pred
        test_speed += time.time() - start

    cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
    #cv2.imshow("Bounding Box", frame)
    #cv2.imshow("Cropped Image", cropped_face)
    cv2.waitKey(1)

    label = np.argmax(prediction)
    score = prediction[0][label] / 2

    print(f"[🧐] Anti-Spoof Result → {'Real' if label == 1 else 'Fake'} | Score: {score:.4f}")
    print(f"[🥚] Raw Scores → Fake2D: {prediction[0][0]:.4f}, Real: {prediction[0][1]:.4f}, Fake3D: {prediction[0][2]:.4f}")

    with open("live_prediction_log.csv", "a") as f:
        csv.writer(f).writerow([time.time(), *prediction[0]])

    return label, score


def extract_eye_tensor(frame, landmarks, indices):
    ih, iw, _ = frame.shape
    x_coords = [int(landmarks[i].x * iw) for i in indices]
    y_coords = [int(landmarks[i].y * ih) for i in indices]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    margin = int((x_max - x_min + y_max - y_min) * 0.4)
    x_min = max(0, x_min - margin)
    x_max = min(iw, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(ih, y_max + margin)
    eye_img = frame[y_min:y_max, x_min:x_max]
    if eye_img.size == 0:
        return None, None
    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_resized = cv2.resize(eye_gray, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    eye_tensor = transform(eye_resized)
    return eye_tensor, eye_resized  # return both tensor and image


def predict_avg_eye_status(left_tensor, right_tensor):
    global last_state
    with torch.no_grad():
        input_tensor = torch.stack([left_tensor, right_tensor]).to("cpu")
        output = blink_model(input_tensor)
        prob = F.softmax(output, dim=1).cpu().numpy()
        avg_prob = np.mean(prob, axis=0)

        print(f"Avg Confidence -> Open: {avg_prob[1]:.2f}, Closed: {avg_prob[0]:.2f}")

        if avg_prob[0] > 0.7:
            pred = 0
        elif avg_prob[1] > 0.7:
            pred = 1
        else:
            pred = last_state

        eye_pred_buffer.append(pred)
        smoothed_pred = round(np.mean(eye_pred_buffer))
        last_state = smoothed_pred
        return smoothed_pred, float(avg_prob[1])


def verify_blink_before_authentication(cap, timeout_sec=10):
    print("[INFO] Blink verification required before authentication...")

    global eye_pred_buffer, last_state
    last_state = 1              # assume "open" at the start
    eye_pred_buffer = deque(maxlen=7)  # smoother blink debounce
    blink_counter = 0
    start_time = time.time()

    while True:
        ret, blink_frame = cap.read()
        if not ret:
            continue

        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print("[⏰] Blink timeout exceeded. Please try again.")
            cv2.destroyWindow("Blink to Authenticate")
            cv2.destroyWindow("👁 Live Left Eye")
            cv2.destroyWindow("👁 Live Right Eye")
            return False

        rgb = cv2.cvtColor(blink_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            ih, iw = blink_frame.shape[:2]
            if not is_face_straight(landmarks, iw, ih):
                cv2.putText(blink_frame, "Face not straight", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("[↩️] Face not straight, ignoring blink.")

            else:
                left_tensor, left_img = extract_eye_tensor(blink_frame, landmarks, LEFT_EYE_LANDMARKS)
                right_tensor, right_img = extract_eye_tensor(blink_frame, landmarks, RIGHT_EYE_LANDMARKS)

                if left_tensor is not None and right_tensor is not None:
                    pred, confidence = predict_avg_eye_status(left_tensor, right_tensor)
                    status = "Open" if pred == 1 else "Closed"

                    if pred == 0 and confidence<0.2:  # eye closed
                        blink_counter += 1
                        print(f"[👁] Blink detected ({blink_counter}/1)")
                    else:
                        blink_counter = 0

                    conf_scalar = float(confidence)  # ensure it's a single float value
                    cv2.putText(blink_frame, f"Blink to Start | Eye: {status} | Conf: {conf_scalar:.2f}",(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            cv2.putText(blink_frame, "Face Not Detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Blink to Authenticate", blink_frame)

        if blink_counter >= 1:
            print("[✅] Blink verified. Waiting for stable frame...")
            time.sleep(1.0)  # give user time to open eyes again

            # Flush 3 frames to ensure clear view
            for _ in range(3):
                cap.read()

            cv2.destroyWindow("Blink to Authenticate")
            cv2.destroyWindow("👁 Live Left Eye")
            cv2.destroyWindow("👁 Live Right Eye")
            return True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[✘] Blink verification cancelled.")
            cv2.destroyWindow("Blink to Authenticate")
            cv2.destroyWindow("👁 Live Left Eye")
            cv2.destroyWindow("👁 Live Right Eye")
            return False


def is_face_straight(landmarks, iw, ih, yaw_thresh=0.1, pitch_thresh=0.1, min_visible_ratio=0.15):
    # Horizontal alignment (yaw): eye centers
    lx = np.mean([landmarks[i].x for i in [33, 133]])
    rx = np.mean([landmarks[i].x for i in [362, 263]])
    eye_diff_x = abs(lx - (1 - rx))

    # Vertical alignment (pitch): nose vs eyes
    nose_y = landmarks[1].y
    eye_line_y = np.mean([landmarks[i].y for i in [33, 133, 362, 263]])
    pitch_angle = nose_y - eye_line_y

    # Face height ratio (to detect lifted head)
    top = eye_line_y
    bottom = landmarks[152].y
    visible_height_ratio = bottom - top

    is_yaw_ok = eye_diff_x < yaw_thresh
    is_pitch_ok = abs(pitch_angle) < pitch_thresh
    is_height_ok = visible_height_ratio > min_visible_ratio

    return is_yaw_ok and is_pitch_ok and is_height_ok

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[INFO] Press 'r' to register | 'a' to authenticate | 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "Press 'r' to register | 'a' to authenticate", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        cv2.imshow("Mask Recognition System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('b'):
            return_to_main = True

        elif key == ord('r'):
            return_to_main = False
            print("[INFO] Registration starting...")

            # Loop until valid name and not already registered
            while True:
                name = input("Enter the user's name: ").strip()
                if not re.fullmatch(r"[A-Za-z ]+", name):
                    print("[!] Invalid name. Only letters and spaces allowed.")
                    continue

                if len(name) < 2:
                    print("[!] Name too short. Please enter a valid full name.")
                    continue

                # Check if both face types already registered
                if name in user_embeddings and \
                "mask" in user_embeddings[name] and \
                "no_mask" in user_embeddings[name]:
                    print(f"[⚠️] User '{name}' is already registered. Please enter a different name.")
                    continue
                break  # valid name

            user_embeddings.setdefault(name, {})

            for mode in ['no_mask', 'mask']:
                embeddings, count = [], 0
                print(f"[INFO] Please {'REMOVE' if mode == 'no_mask' else 'WEAR'} your mask.")

                for c in [3, 2, 1]:
                    temp = frame.copy()
                    cv2.putText(temp, f"Starting {mode} capture in {c}...", (80, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
                    cv2.imshow("Mask Recognition System", temp)
                    cv2.waitKey(1000)

                while count < 5:
                    ret, frame = cap.read()
                    if not ret: continue
                    faces = detect_faces_retinaface(frame)
                    if not faces: continue
                    (x1, y1, x2, y2) = faces[0]
                    label, _ = anti_spoof_check(frame)
                    if label != 1:
                        print("[✘] Spoof detected. Try again.")
                        continue
                    face_crop = frame[y1:y2, x1:x2]

                    # Detect mask state to verify correctness
                    resized = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (224, 224))
                    arr = preprocess_input(img_to_array(resized)[None, ...])
                    mask_prob, no_mask_prob = mask_model.predict(arr)[0]
                    is_masked = mask_prob > no_mask_prob
                    current_mode = "mask" if is_masked else "no_mask"

                    if current_mode != mode:
                        print(f"[✘] Incorrect face type. Expected: {mode}. Detected: {current_mode}.")
                        continue

                    # Check if face already exists
                    current_embedding = extract_embedding(face_crop)
                    identity = recognize_user(current_embedding, is_masked)
                    if identity != "Unknown":
                        print(f"[⚠️] This face is already registered as '{identity}'. Returning to main screen.")
                        return_to_main = True
                        break

                    embeddings.append(current_embedding)
                    count += 1

                if count < 5:
                    print(f"[!] Registration for {mode} face cancelled.")
                    return_to_main = True
                    break

                user_embeddings[name][mode] = np.mean(np.stack(embeddings), axis=0)
                print(f"[✔] {name}'s {mode.upper()} face registered.")

            if 'return_to_main' in locals() and return_to_main:
                print(f"[❌] Registration for '{name}' failed or cancelled.")
                continue  # back to main screen without saving

            np.save("user_embeddings.npy", user_embeddings)
            print(f"[✅] Full registration for '{name}' completed.")


        elif key == ord('i'):
            if not verify_blink_before_authentication(cap):
                continue  # Skip if user quits during blink phase

            # Proceed with your original authentication logic
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame capture failed!")
                continue
            faces = detect_faces_retinaface(frame)
            if not faces:
                print("[!] No face detected.")
                continue
            (x1, y1, x2, y2) = faces[0]
            label, _ = anti_spoof_check(frame)
            if label != 1:
                print("[✘] Spoof detected! Access denied.")
                continue

            face_crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (224, 224))
            arr = preprocess_input(img_to_array(resized)[None, ...])
            mask_prob, no_mask_prob = mask_model.predict(arr)[0]
            is_masked = mask_prob > no_mask_prob
            mask_label = "Mask" if is_masked else "No Mask"

            current_embedding = extract_embedding(face_crop)
            identity = recognize_user(current_embedding, is_masked)
            if identity == "Unknown":
                print("[❌] User not recognized. Access denied.")
                continue
            else:
                print(f"[✅] Welcome {identity}. Access granted.")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([identity, timestamp, "Real", mask_label])
            print(f"[✔] {identity} recognized with [{mask_label}] at {timestamp}")
        
        elif key == ord('o'):
            if not verify_blink_before_authentication(cap):
                continue  # Skip if user quits during blink phase

            # Proceed with your original authentication logic
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame capture failed!")
                continue
            faces = detect_faces_retinaface(frame)
            if not faces:
                print("[!] No face detected.")
                continue
            (x1, y1, x2, y2) = faces[0]
            label, _ = anti_spoof_check(frame)
            if label != 1:
                print("[✘] Spoof detected! Access denied.")
                continue

            face_crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (224, 224))
            arr = preprocess_input(img_to_array(resized)[None, ...])
            mask_prob, no_mask_prob = mask_model.predict(arr)[0]
            is_masked = mask_prob > no_mask_prob
            mask_label = "Mask" if is_masked else "No Mask"

            current_embedding = extract_embedding(face_crop)
            identity = recognize_user(current_embedding, is_masked)
            if identity == "Unknown":
                print("[❌] User not recognized. Access denied.")
                continue
            else:
                print(f"[✅] Bye {identity}. Access granted.")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([identity, timestamp, "Real", mask_label])
            print(f"[✔] {identity} recognized with [{mask_label}] at {timestamp}")


    cap.release()
    cv2.destroyAllWindows()