import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

def is_open_hand(landmarks):
    fingers = [landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]]
    thumb = landmarks[4].x > landmarks[3].x
    return thumb and all(fingers)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            if is_open_hand(hand.landmark):
                h, w = frame.shape[:2]
                cx, cy = int(hand.landmark[9].x * w), int(hand.landmark[9].y * h)
                cv2.circle(frame, (cx, cy), 100, (0, 165, 255), 4)  # Orange shield
                cv2.putText(frame, "Mandala Shield", (cx - 80, cy - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Mandala Shield", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
