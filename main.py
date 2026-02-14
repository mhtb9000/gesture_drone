import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import math

model = tf.keras.models.load_model("model.h5")
labels = ["down","left","right","up"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

neutral_d = None
calib = []

prev_cmd = "FLOAT"
stable = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    command = "FLOAT"
    conf = 0

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,hand,
            mp_hands.HAND_CONNECTIONS)

            h,w,_ = frame.shape

            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            x1,y1 = int(min(xs)*w)-40,int(min(ys)*h)-40
            x2,y2 = int(max(xs)*w)+40,int(max(ys)*h)+40

            x1=max(0,x1)
            y1=max(0,y1)
            x2=min(w,x2)
            y2=min(h,y2)

            hand_img = frame[y1:y2,x1:x2]

            if hand_img.size != 0:

                h_img,w_img,_=hand_img.shape
                size=max(h_img,w_img)

                square=255*np.ones((size,size,3),dtype=np.uint8)

                y_offset=(size-h_img)//2
                x_offset=(size-w_img)//2

                square[y_offset:y_offset+h_img,
                x_offset:x_offset+w_img]=hand_img

                img=cv2.resize(square,(160,160))
                img=img/255.0
                img=np.expand_dims(img,axis=0)

                pred=model.predict(img,verbose=0)[0]
                idx=np.argmax(pred)
                conf=pred[idx]
                cnn_cmd=labels[idx]

                thumb=hand.landmark[4]
                index=hand.landmark[8]

                d=math.sqrt(
                    (thumb.x-index.x)**2+
                    (thumb.y-index.y)**2
                )

                if neutral_d is None:

                    calib.append(d)

                    if len(calib)<20:
                        cv2.putText(frame,
                        "Calibrating...",
                        (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,(0,0,255),2)
                        continue

                    neutral_d=sum(calib)/len(calib)

                delta=d-neutral_d
                speed=np.clip(delta*500,-100,100)

                if speed>20:
                    raw="FORWARD"
                elif speed<-20:
                    raw="BACKWARD"
                else:
                    if conf>0.75:
                        raw=cnn_cmd.upper()
                    else:
                        raw="FLOAT"

                if raw==prev_cmd:
                    stable+=1
                else:
                    stable=0

                if stable<3:
                    command="FLOAT"
                else:
                    command=raw

                prev_cmd=raw

    cv2.putText(frame,
    f"{command} {conf*100:.1f}%",
    (20,50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,(0,255,0),2)

    print(command)

    cv2.imshow("Drone Control",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()