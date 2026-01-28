import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import mediapipe as mp
import torchvision.transforms as transforms
from retinaface import RetinaFace
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.model import Model as BlinkModel
from collections import deque
from scipy.spatial.distance import cosine

# === Load Models ===
mask_model = load_model("/Users/calebyeo/Downloads/4. Degree 3/Project/System/models/mask_detection_model/mask_detector.model.h5")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recognition_model = torch.load("/Users/calebyeo/Downloads/4. Degree 3/Project/System/models/mask_recognition_model/InceptionResNetV1_ArcFace.pt", map_location=device, weights_only=False)
recognition_model.eval().to(device)
model_test = AntiSpoofPredict(0)
cropper = CropImage()
anti_spoof_model_dir = "models/anti_spoof_model"
RECOGNITION_THRESHOLD = 0.75

# === Blink Detection ===
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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]


MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

LANDMARK_INDEXES = [1, 152, 33, 263, 78, 308]  # Nose, chin, eyes, mouth
alignment_buffer = deque(maxlen=5)


def detect_faces_retinaface(frame):
    detections = RetinaFace.detect_faces(frame)
    faces = [det["facial_area"] for det in detections.values()] if isinstance(detections, dict) else []
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


def extract_eye_tensor(frame, landmarks, indices):
    ih, iw, _ = frame.shape
    x_coords = [int(landmarks[i].x * iw) for i in indices]
    y_coords = [int(landmarks[i].y * ih) for i in indices]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    margin = int((x_max - x_min + y_max - y_min) * 0.2)
    x_min = max(0, x_min - margin)
    x_max = min(iw, x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(ih, y_max + margin)
    eye_img = frame[y_min:y_max, x_min:x_max]
    if eye_img.size == 0:
        return None, None
    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_sharp = cv2.filter2D(eye_gray, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    eye_resized = cv2.resize(eye_gray, (EYE_IMG_SIZE, EYE_IMG_SIZE))
    eye_tensor = transform(eye_resized)
    return eye_tensor, eye_resized


def predict_avg_eye_status(left_tensor, right_tensor):
    global eye_pred_buffer, last_state
    with torch.no_grad():
        input_tensor = torch.stack([left_tensor, right_tensor]).to("cpu")
        output = blink_model(input_tensor)
        prob = F.softmax(output, dim=1).cpu().numpy()
        avg_prob = np.mean(prob, axis=0)

        if avg_prob[0] > 0.7:
            pred = 0  # Closed
        elif avg_prob[1] > 0.7:
            pred = 1  # Open
        else:
            pred = last_state

        eye_pred_buffer.append(pred)
        smoothed_pred = round(np.mean(eye_pred_buffer))
        last_state = smoothed_pred
        return smoothed_pred, float(avg_prob[1]), float(avg_prob[0])  # open_conf, closed_conf


def get_camera_matrix(width, height):
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1)) 
    return camera_matrix, dist_coeffs


def is_face_straight_pnp(landmarks, iw, ih, yaw_thresh=35, pitch_thresh=35, roll_thresh=45):
    image_points = np.array([
        [landmarks[i].x * iw, landmarks[i].y * ih] for i in LANDMARK_INDEXES
    ], dtype=np.float64)

    camera_matrix, dist_coeffs = get_camera_matrix(iw, ih)

    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        print("[PnP] solvePnP failed.")
        return False

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rot_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, roll = euler_angles.flatten()

    yaw = (yaw + 180) % 360 - 180
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180

    roll = (roll + 180) % 360 - 180
    print(f"[YAW={yaw:.2f}] [PITCH={pitch:.2f}] [ROLL={roll:.2f}]")

    if abs(roll) > 170:
        print("[INFO] Ignoring extreme roll caused by PnP wraparound.")
        roll = 0

    return abs(yaw) < yaw_thresh and abs(pitch) < pitch_thresh and abs(roll) < roll_thresh


def recognize_user(current_embedding, is_masked, db_users, threshold=RECOGNITION_THRESHOLD):
    key = "mask" if is_masked else "no_mask"
    max_sim, identity = -1, "Unknown"
    for user in db_users:
        db_emb = user.get(key)
        if db_emb is None:
            continue
        sim = np.dot(current_embedding, db_emb) / (np.linalg.norm(current_embedding) * np.linalg.norm(db_emb))
        if sim > max_sim:
            max_sim, identity = sim, user["name"]
    return identity if max_sim >= threshold else "Unknown"


def detect_mask_status(face_crop):
    resized = cv2.resize(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), (224, 224))
    arr = preprocess_input(img_to_array(resized)[None, ...])
    mask_prob, no_mask_prob = mask_model.predict(arr)[0]
    return (mask_prob > no_mask_prob), ("Mask" if mask_prob > no_mask_prob else "No Mask")


def anti_spoof_check(frame):
    detections = RetinaFace.detect_faces(frame)
    
    if not isinstance(detections, dict) or len(detections) == 0:
        return 0, 0.0
    
    facial_area = list(detections.values())[0]["facial_area"]
    x1, y1, x2, y2 = facial_area
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    target_h = y2 - y1
    target_w = int(target_h * 3 / 4)
    new_x1 = max(0, cx - target_w // 2)
    new_y1 = max(0, cy - target_h // 2)
    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h
    cropped_face = frame[new_y1:new_y2, new_x1:new_x2]
    bbox = model_test.get_bbox(cropped_face)
    prediction = np.zeros((1, 3))

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
        pred = model_test.predict(img, os.path.join(anti_spoof_model_dir, model_name))
        prediction += pred

    label = np.argmax(prediction)
    score = prediction[0][label] / 2
    print(f"[🧐] Anti-Spoof Result → {'Real' if label == 1 else 'Fake'} | Score: {score:.4f}")
    print(f"[🥚] Raw Scores → Fake2D: {prediction[0][0]:.4f}, Real: {prediction[0][1]:.4f}, Fake3D: {prediction[0][2]:.4f}")
    return label, score