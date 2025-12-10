from ultralytics import YOLO
import cv2
import os
import time

# Load models
face_detector = YOLO("yolov8n-face.pt")  # face detection model
emotion_model = YOLO("runs/classify/train/weights/best(3).pt")  # emotion classifier 100 epoch (3) if 50 epoch use (2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# save cropped human faces into this directory
SAVE_DIR = "human_feedback/"
os.makedirs(SAVE_DIR, exist_ok=True)

EMOTIONS = list(emotion_model.names.values())

for label in EMOTIONS:
    os.makedirs(f"{SAVE_DIR}/{label}", exist_ok=True)

# function to save cropped face with timestamp
def save_face(face_img, label):
    timestamp = str(int(time.time() * 1000))
    path = f"{SAVE_DIR}/{label}/{timestamp}.jpg"
    cv2.imwrite(path, face_img)
    print(f"Saved: {path}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    last_face = None
    last_label = None

    # Detect faces
    face_results = face_detector(frame, verbose=False)

    for box in face_results[0].boxes:
        if box.conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Resize for classifier
        face_resized = cv2.resize(face, (224, 224))

        # Run emotion detection
        emotion_results = emotion_model(face_resized, verbose=False)
        if not hasattr(emotion_results[0], "probs"):
            continue

        cls_id = int(emotion_results[0].probs.top1)
        label = emotion_results[0].names[cls_id]
        confidence = float(emotion_results[0].probs.top1conf)

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Store last detected face for potential saving
        last_face = face_resized
        last_label = label

    cv2.imshow("Emotion Recognition", frame)
    # Handle keypress once per frame
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('u') and last_face is not None:
        save_face(last_face, last_label)
        print(f"User confirmed label: {last_label}")
    elif key == ord('x') and last_face is not None:
        print("Model predicted:", last_label)
        print("Choose correct label:")
        for i, name in enumerate(EMOTIONS):
            print(f"{i+1} = {name}")
    elif key in [ord(str(i)) for i in range(1, len(EMOTIONS)+1)] and last_face is not None:
        correct_label = EMOTIONS[key - ord('1')]
        save_face(last_face, correct_label)


cap.release()
cv2.destroyAllWindows()
