import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# LOAD MODEL
# ===============================
model = YOLO(
    "https://huggingface.co/ParkVerc/model_s_crtami/resolve/main/best.pt"
)

PARKING_LINE_CLASS_ID = 7

# ===============================
# PARAMETRI
# ===============================
MAX_OFFSET = 150
MAX_ROTATION = 540

ACTIVE_ZONE_HALF_WIDTH = 180   # ±100 px → VOLAN RAVEN

DEAD_ZONE = 8
MIN_STEP = 2
MAX_STEP = 20

CONFIDENCE_MAX = 5
CONFIDENCE_THRESHOLD = 3
SAME_LINE_TOL = 60


# ===============================
# OFFSET → ROTACIJA
# ===============================
def offset_to_rotation(offset, max_offset=150, max_rotation=540):
    offset = np.clip(offset, -max_offset, max_offset)
    return (offset / max_offset) * max_rotation


# ===============================
# MAIN
# ===============================
def main():
    cap = cv2.VideoCapture("IMG_4905.mp4")

    current_offset = 0
    last_candidate_x = None
    confidence = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        ref_x = w // 2

        result = model(frame, verbose=False, conf=0.6)[0]

        candidates = []

        # ===============================
        # YOLO OBB DETEKCIJE
        # ===============================
        if result.obb is not None:
            for i, obb in enumerate(result.obb.xywhr):
                cls_id = int(result.obb.cls[i].item())
                if cls_id != PARKING_LINE_CLASS_ID:
                    continue

                x, y, width, height, angle = [
                    obb[j].item() for j in range(5)
                ]

                candidates.append((x, y))

                # 🟩 OBB debug
                box = ((x, y), (width, height), angle * 180 / np.pi)
                pts = cv2.boxPoints(box)
                pts = np.intp(pts)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        # ===============================
        # IZBIRA CILJNE ČRTE (OBRATNA LOGIKA)
        # ===============================
        target_offset = 0

        if candidates:
            candidate_x, candidate_y = min(
                candidates,
                key=lambda p: abs(p[0] - ref_x) + 0.3 * abs(p[1] - h)
            )

            if last_candidate_x is None or abs(candidate_x - last_candidate_x) < SAME_LINE_TOL:
                confidence = min(CONFIDENCE_MAX, confidence + 1)
            else:
                confidence = max(0, confidence - 1)

            last_candidate_x = candidate_x

            if confidence >= CONFIDENCE_THRESHOLD:

                # 🔁 KLJUČ: ACTIVE ZONE = NO STEERING
                if abs(candidate_x - ref_x) <= ACTIVE_ZONE_HALF_WIDTH:
                    target_offset = 0
                else:
                    target_offset = candidate_x - ref_x
                    target_offset = np.clip(target_offset, -MAX_OFFSET, MAX_OFFSET)

        else:
            target_offset = 0
            confidence = 0

        # ===============================
        # DEAD ZONE (še dodatna stabilnost)
        # ===============================
        if abs(target_offset) < DEAD_ZONE:
            target_offset = 0

        # ===============================
        # DINAMIČNI STEP + RAMPANJE
        # ===============================
        error = target_offset - current_offset
        step = int(np.clip(abs(error) * 0.2, MIN_STEP, MAX_STEP))

        if current_offset < target_offset:
            current_offset = min(current_offset + step, target_offset)
        elif current_offset > target_offset:
            current_offset = max(current_offset - step, target_offset)

        offset = int(current_offset)
        rotation = offset_to_rotation(offset, MAX_OFFSET, MAX_ROTATION)

        # ===============================
        # RISANJE
        # ===============================

        # 🟨 SREDINA
        cv2.line(frame, (ref_x, 0), (ref_x, h), (255, 255, 0), 2)

        # 🔴 ACTIVE ZONE (NO STEERING)
        cv2.line(frame, (ref_x - ACTIVE_ZONE_HALF_WIDTH, 0),
                 (ref_x - ACTIVE_ZONE_HALF_WIDTH, h), (0, 0, 255), 2)
        cv2.line(frame, (ref_x + ACTIVE_ZONE_HALF_WIDTH, 0),
                 (ref_x + ACTIVE_ZONE_HALF_WIDTH, h), (0, 0, 255), 2)

        cv2.putText(frame, "NO STEERING ZONE",
                    (ref_x - ACTIVE_ZONE_HALF_WIDTH + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 🟦 uporabljena črta
        chosen_x = int(ref_x + offset)
        cv2.line(frame, (chosen_x, 0), (chosen_x, h), (255, 0, 0), 2)

        # 🔴 puščica
        cv2.arrowedLine(
            frame,
            (ref_x, h // 2),
            (chosen_x, h // 2),
            (0, 0, 255),
            3,
            tipLength=0.15
        )

        # ===============================
        # TEKST
        # ===============================
        cv2.putText(frame, f"OFFSET: {offset}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.putText(frame, f"ROTATION: {rotation:.1f} deg", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 120, 0), 3)

        cv2.putText(frame, f"CONFIDENCE: {confidence}", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        cv2.imshow("PARK ASSIST – OUTSIDE ZONE STEERING", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
