import cv2
from ultralytics import YOLO

# kleines, schnelles Modell
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)  # ggf. 1/2 falls deine Cam woanders liegt

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Inference
    results = model(frame, verbose=False)[0]

    # nur "person" (class 0)
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Tankstellen Demo - Person Detection", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()