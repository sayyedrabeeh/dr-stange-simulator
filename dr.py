import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def get_shield_radius(z):
    """Convert depth (z) to shield radius — closer hand = bigger shield"""
    return int(200 * (1 - min(max(z, 0.0), 0.5))) + 30  

def get_hand_center(landmarks, w, h):
    return int(landmarks[9].x * w), int(landmarks[9].y * h), landmarks[9].z

def draw_shield(img, center, radius, color=(0, 150, 255)):
    cv2.circle(img, center, radius, color, 3, lineType=cv2.LINE_AA)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_centers = []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            cx, cy, cz = get_hand_center(hand.landmark, w, h)
            hand_centers.append(((cx, cy), cz))

    if len(hand_centers) == 2:
        
        dist_between = math.hypot(
            hand_centers[0][0][0] - hand_centers[1][0][0],
            hand_centers[0][0][1] - hand_centers[1][0][1]
        )

        if dist_between < 200:
           
            mid = (
                (hand_centers[0][0][0] + hand_centers[1][0][0]) // 2,
                (hand_centers[0][0][1] + hand_centers[1][0][1]) // 2
            )
            avg_z = (hand_centers[0][1] + hand_centers[1][1]) / 2
            radius = get_shield_radius(avg_z) + 50
            draw_shield(frame, mid, radius, color=(0, 255, 200))
            cv2.putText(frame, "Unified Shield", (mid[0] - 80, mid[1] - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        else:
            
            for center, z in hand_centers:
                radius = get_shield_radius(z)
                draw_shield(frame, center, radius)
                cv2.putText(frame, "Shield", (center[0] - 40, center[1] - radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    elif len(hand_centers) == 1:
        center, z = hand_centers[0]
        radius = get_shield_radius(z)
        draw_shield(frame, center, radius)
        cv2.putText(frame, "Shield", (center[0] - 40, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Dynamic Mandala Shields", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
