import cv2
import mediapipe as mp
import os
import time
import numpy as np

GESTURE = "right"                 
SAVE_DIR = f"data/raw/{GESTURE}"
TARGET = 350
DELAY = 0.1                  #   DELAY SET 

os.makedirs(SAVE_DIR, exist_ok=True)


existing = len([
    f for f in os.listdir(SAVE_DIR)
    if f.endswith(".jpg")
])

count = existing

if count >= TARGET:
    print("Already reached 350 images.")
    exit()

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands.Hands()

last_capture = time.time()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            h, w, _ = frame.shape

            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]


            x1 = int(min(xs) * w) - 40
            y1 = int(min(ys) * h) - 40
            x2 = int(max(xs) * w) + 40
            y2 = int(max(ys) * h) + 40

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)


            hand_img = frame[y1:y2, x1:x2]

            if hand_img.size != 0:

                if time.time() - last_capture > DELAY:

                    if count < TARGET:

                        # --------- SQUARE PAD (NO DISTORTION) ----------
                        h_img, w_img, _ = hand_img.shape
                        size = max(h_img, w_img)

                        square = 255 * np.ones(
                            (size, size, 3),
                            dtype=np.uint8
                        )

                        y_offset = (size - h_img) // 2
                        x_offset = (size - w_img) // 2

                        square[
                        y_offset:y_offset+h_img,
                        x_offset:x_offset+w_img
                        ] = hand_img
                        

                        # -------- RESIZE AFTER PADDING --------
                        hand_img = cv2.resize(square, (128,128))
                        

                        cv2.imwrite(
                        f"{SAVE_DIR}/{count}.jpg",
                        hand_img
                        )

                        count += 1
                        last_capture = time.time()

    cv2.putText(
    frame,
    f"{GESTURE.upper()} : {count}/{TARGET}",
    (20,40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,(0,255,0),2
    )

    cv2.imshow("Capture", frame)

    if count >= TARGET:
        print("Reached 350 images")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()