import cv2
import mediapipe as mp
import numpy as np
from subprocess import call

mpHands = mp.solutions.hands
hands = mpHands.Hands(False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
capture_size = (int(cap.get(3)), int(cap.get(4)))

while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,landmark,mpHands.HAND_CONNECTIONS)

        detected_hands = results.multi_hand_landmarks
        for hand in detected_hands:
            for id, landm in enumerate(hand.landmark):
                h,w,c = img.shape

                cx, cy = int(landm.x * w), int(
                    landm.y * h)
                if id == 4:
                    x1, y1 = cx, cy
                if id == 8:
                    x2,y2 = cx,cy
                    if x1 and y1 and y2 and x2:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(img, (x1, y1), 15, (255, 0, 0), -1)
                        cv2.circle(img, (x2, y2), 15, (255, 0, 0), -1)
                        cv2.circle(img, (cx, cy), 15, (255, 0, 0), -1)
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                        if distance < 50:
                            cv2.circle(img, (cx, cy), 15, (0, 255, 0), -1)
                        if distance > 300:
                            cv2.circle(img, (x1, y1), 15, (0, 0, 255), -1)
                            cv2.circle(img, (x2, y2), 15, (0, 0, 255), -1)

                        vol = np.interp(distance, [50, 300], [0, 100])
                        vol = int(vol / 20) * 20
                        cv2.putText(img, str(round(vol, 2)), (x1 + 40, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

                        cv2.rectangle(img, (50, 100), (85, 200), (24, 255, 120), 3)
                        cv2.rectangle(img, (50, 200 - vol), (85, 200), (29, 255, 248), -1)

                        if vol == 100:
                            cv2.rectangle(img, (50, 200 - vol), (85, 200), (228, 88, 88), -1)
                        if vol == 0:
                            cv2.rectangle(img, (50, 200 - vol), (85, 200), (0, 0, 0), -1)

                        call([f"osascript -e 'set volume output volume {vol}'"], shell=True)

    cv2.imshow("Video",img)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
