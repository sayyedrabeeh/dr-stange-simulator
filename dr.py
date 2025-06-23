import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def get_shield_radius(z):
    return int(200 * (1 - min(max(z, 0.0), 0.5))) + 40

def get_hand_center(landmarks, w, h):
    return int(landmarks[9].x * w), int(landmarks[9].y * h), landmarks[9].z

def draw_mandala_shield(img, center, radius, rotation=0):
    overlay = img.copy()

    
    for i in range(3):
        glow_radius = radius + i * 2
        alpha = 0.3 - i * 0.08
        color = (0, int(150 * alpha * 5), int(255 * alpha * 5))
        cv2.circle(overlay, center, glow_radius, color, 2, cv2.LINE_AA)

   
    num_lines = 16
    for i in range(num_lines):
        angle = (2 * math.pi * i / num_lines) + rotation
        x1 = int(center[0] + (radius - 10) * math.cos(angle))
        y1 = int(center[1] + (radius - 10) * math.sin(angle))
        x2 = int(center[0] + (radius + 10) * math.cos(angle))
        y2 = int(center[1] + (radius + 10) * math.sin(angle))
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

     
    cv2.circle(overlay, center, int(radius * 0.6), (255, 200, 100), 1, cv2.LINE_AA)

     
    for i in range(8):
        angle = (2 * math.pi * i / 8) + rotation * 0.5
        petal_radius = int(radius * 0.7)
        x = int(center[0] + petal_radius * math.cos(angle))
        y = int(center[1] + petal_radius * math.sin(angle))
        cv2.line(overlay, center, (x, y), (255, 255, 100), 1, cv2.LINE_AA)

    
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_centers = []
    rotation = time.time() * 1.5  

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            cx, cy, cz = get_hand_center(hand.landmark, w, h)
            hand_centers.append(((cx, cy), cz))

    if len(hand_centers) == 2:
        dist = math.hypot(
            hand_centers[0][0][0] - hand_centers[1][0][0],
            hand_centers[0][0][1] - hand_centers[1][0][1]
        )
        if dist < 200:
            mid = (
                (hand_centers[0][0][0] + hand_centers[1][0][0]) // 2,
                (hand_centers[0][0][1] + hand_centers[1][0][1]) // 2
            )
            avg_z = (hand_centers[0][1] + hand_centers[1][1]) / 2
            radius = get_shield_radius(avg_z) + 50
            draw_mandala_shield(frame, mid, radius, rotation)
            cv2.putText(frame, "Unified Shield", (mid[0] - 80, mid[1] - radius - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        else:
            for center, z in hand_centers:
                radius = get_shield_radius(z)
                draw_mandala_shield(frame, center, radius, rotation)
    elif len(hand_centers) == 1:
        center, z = hand_centers[0]
        radius = get_shield_radius(z)
        draw_mandala_shield(frame, center, radius, rotation)

    cv2.imshow("Dr. Strange Mandala Shield", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
