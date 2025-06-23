import cv2
import mediapipe as mp
import numpy as np
import math
import time

 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
def get_shield_radius(z):
    z = min(max(z, 0.0), 0.4)
    return int(100 + (1 - z / 0.4) * 150)   

def get_hand_center(landmarks, w, h):
    return int(landmarks[9].x * w), int(landmarks[9].y * h), landmarks[9].z

 
def draw_tao_mandala(img, center, radius, rotation=0):
    overlay = img.copy()

    
    for i in range(3):
        cv2.circle(overlay, center, radius + i*3, (30, 200, 255), 2, cv2.LINE_AA)

   
    num_lines = 12
    for i in range(num_lines):
        angle = 2 * math.pi * i / num_lines + rotation
        x1 = int(center[0] + (radius - 20) * math.cos(angle))
        y1 = int(center[1] + (radius - 20) * math.sin(angle))
        x2 = int(center[0] + (radius + 20) * math.cos(angle))
        y2 = int(center[1] + (radius + 20) * math.sin(angle))
        cv2.line(overlay, (x1, y1), (x2, y2), (255, 200, 100), 1, cv2.LINE_AA)

 
    for angle in [0, math.pi/4, math.pi/2]:
        offset = int(radius * 0.6)
        pts = []
        for i in range(4):
            a = angle + i * math.pi / 2
            x = int(center[0] + offset * math.cos(a))
            y = int(center[1] + offset * math.sin(a))
            pts.append((x, y))
        cv2.polylines(overlay, [np.array(pts)], isClosed=True, color=(255, 180, 0), thickness=1)

    
    alpha = 0.7
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
    rotation = time.time() * 2

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            cx, cy, cz = get_hand_center(hand.landmark, w, h)
            hand_centers.append(((cx, cy), cz))

    if len(hand_centers) == 2:
        dist = math.hypot(
            hand_centers[0][0][0] - hand_centers[1][0][0],
            hand_centers[0][0][1] - hand_centers[1][0][1]
        )
        if dist < 250:
            
            mid = (
                (hand_centers[0][0][0] + hand_centers[1][0][0]) // 2,
                (hand_centers[0][0][1] + hand_centers[1][0][1]) // 2
            )
            avg_z = (hand_centers[0][1] + hand_centers[1][1]) / 2
            radius = get_shield_radius(avg_z) + 40
            draw_tao_mandala(frame, mid, radius, rotation)
            cv2.putText(frame, "Unified Shield", (mid[0] - 80, mid[1] - radius - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        else:
             
            for center, z in hand_centers:
                radius = get_shield_radius(z)
                draw_tao_mandala(frame, center, radius, rotation)
    elif len(hand_centers) == 1:
        center, z = hand_centers[0]
        radius = get_shield_radius(z)
        draw_tao_mandala(frame, center, radius, rotation)

    cv2.imshow("Dr. Strange Mandala Shield", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
