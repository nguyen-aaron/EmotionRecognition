from ultralytics import YOLO
import cv2

# Load models
face_detector = YOLO("yolov8n-face.pt")  # face detection model
emotion_model = YOLO("runs/classify/train/weights/best.pt")  # emotion classifier

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
