import time
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# ----------------------------
# Konfiguration
# ----------------------------
# Kamera-Index: 0 = eingebaute Mac-Kamera, 1/2 = oft externe/virtuelle (z.B. Iriun)
CAMERA_INDEX = 0

# Zonen (x1, y1, x2, y2) im Bild. Diese Werte musst du einmal passend einstellen.
# Tipp: Erst laufen lassen, dann mit den angezeigten Rechtecken anpassen.
SHELF_ZONE = (40, 120, 300, 420)   # "Regal" / Ware-Bereich
CASH_ZONE  = (330, 120, 560, 420)  # "Kasse" / Bezahlen
EXIT_ZONE  = (580, 120, 760, 420)  # "Ausgang" / Tür

# Wie lange wir einen Track ohne neue Frames behalten (Sekunden)
TRACK_TTL_SECONDS = 3.0

# Cooldown, damit nicht jede Frame-Spur spammt (Sekunden)
ALERT_COOLDOWN_SECONDS = 5.0


# ----------------------------
# Hilfsfunktionen
# ----------------------------

def point_in_rect(px: int, py: int, rect) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_zone(frame, rect, label: str):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(frame, label, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


@dataclass
class TrackState:
    last_seen: float
    seen_shelf: bool = False
    seen_cash: bool = False
    seen_exit: bool = False
    last_alert: float = 0.0


# ----------------------------
# Modell + Kamera
# ----------------------------

model = YOLO("yolov8n.pt")  # kleines, schnelles Modell
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(f"Kamera konnte nicht geöffnet werden (Index {CAMERA_INDEX}).")

# Track-Zustände nach ID
tracks: dict[int, TrackState] = {}


# ----------------------------
# Loop
# ----------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # --- Personen-Tracking (liefert IDs) ---
    # persist=True sorgt dafür, dass IDs über Frames hinweg stabiler bleiben
    # classes=[0] -> nur Person
    results = model.track(frame, persist=True, verbose=False, classes=[0])[0]

    # Zonen zeichnen
    draw_zone(frame, SHELF_ZONE, "REGAL")
    draw_zone(frame, CASH_ZONE, "KASSE")
    draw_zone(frame, EXIT_ZONE, "AUSGANG")

    alert_this_frame = False
    alert_text = ""

    # Ergebnisse können leer sein
    boxes = getattr(results, "boxes", None)
    if boxes is not None and boxes.xyxy is not None and len(boxes) > 0:
        xyxy = boxes.xyxy
        confs = boxes.conf
        # IDs kommen von tracker, können am Anfang None sein
        ids = boxes.id

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(int, xyxy[i].tolist())
            conf = float(confs[i]) if confs is not None else 0.0

            track_id = None
            if ids is not None:
                # ids[i] ist ein Tensor/NumPy-Skalar
                try:
                    track_id = int(ids[i].item())
                except Exception:
                    track_id = None

            # Mittelpunkt der Person
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Bounding Box + Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"person {conf:.2f}"
            if track_id is not None:
                label = f"ID {track_id}  {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            # Track-State updaten
            if track_id is not None:
                st = tracks.get(track_id)
                if st is None:
                    st = TrackState(last_seen=now)
                    tracks[track_id] = st
                st.last_seen = now

                # Zonen-Flags
                if point_in_rect(cx, cy, SHELF_ZONE):
                    st.seen_shelf = True
                if point_in_rect(cx, cy, CASH_ZONE):
                    st.seen_cash = True
                if point_in_rect(cx, cy, EXIT_ZONE):
                    st.seen_exit = True

                # ALERT-Logik (rechtlich sauber: "Vorgang prüfen", keine Schuldzuweisung)
                # Regal -> Ausgang, ohne Kasse
                in_exit_now = point_in_rect(cx, cy, EXIT_ZONE)
                if in_exit_now and st.seen_shelf and not st.seen_cash:
                    if (now - st.last_alert) >= ALERT_COOLDOWN_SECONDS:
                        st.last_alert = now
                        alert_this_frame = True
                        alert_text = f"⚠️ Vorgang prüfen (ID {track_id})"
                        print(alert_text)

    # Alte Tracks aufräumen
    to_delete = [tid for tid, st in tracks.items() if (now - st.last_seen) > TRACK_TTL_SECONDS]
    for tid in to_delete:
        del tracks[tid]

    # Alert-Overlay
    if alert_this_frame:
        cv2.putText(frame, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Tankstellen Demo - Person Tracking + Zonen", frame)

    # ESC zum Beenden
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

cap.release()
cv2.destroyAllWindows()