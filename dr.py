import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

 
mandala = Image.open("ab.png").convert("RGBA")
mandala_np = np.array(mandala)

def overlay_image_alpha(bg, fg, x, y):
    """Overlay fg on bg at (x,y) with alpha blending"""
    h, w = fg.shape[:2]
    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg   

    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] +
                               alpha_bg * bg[y:y+h, x:x+w, c])
    return bg

def get_shield(mandala_base, scale):
    size = int(100 * scale) + 100
    resized = cv2.resize(mandala_base, (size, size), interpolation=cv2.INTER_AREA)
    return resized

def get_center_and_depth(hand_landmarks, w, h):
    cx = int(hand_landmarks[9].x * w)
    cy = int(hand_landmarks[9].y * h)
    cz = hand_landmarks[9].z
    return (cx, cy), cz

 
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

 
mandala_cv = cv2.cvtColor(np.array(mandala), cv2.COLOR_RGBA2BGRA)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    centers = []
    depths = []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            center, depth = get_center_and_depth(hand.landmark, w, h)
            centers.append(center)
            depths.append(depth)

    if len(centers) == 2:
        dist = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
        if dist < 200:
            
            mid = ((centers[0][0] + centers[1][0]) // 2,
                   (centers[0][1] + centers[1][1]) // 2)
            avg_depth = (depths[0] + depths[1]) / 2
            scale = max(0.1, min(1.0, 1 - avg_depth * 2))
            shield = get_shield(mandala_cv, scale * 1.4)
            x, y = mid[0] - shield.shape[1] // 2, mid[1] - shield.shape[0] // 2
            frame = overlay_image_alpha(frame, shield, x, y)
            cv2.putText(frame, "Unified Shield", (mid[0] - 80, mid[1] - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            
            for i in range(2):
                scale = max(0.1, min(1.0, 1 - depths[i] * 2))
                shield = get_shield(mandala_cv, scale)
                x, y = centers[i][0] - shield.shape[1] // 2, centers[i][1] - shield.shape[0] // 2
                frame = overlay_image_alpha(frame, shield, x, y)
    elif len(centers) == 1:
        scale = max(0.1, min(1.0, 1 - depths[0] * 2))
        shield = get_shield(mandala_cv, scale)
        x, y = centers[0][0] - shield.shape[1] // 2, centers[0][1] - shield.shape[0] // 2
        frame = overlay_image_alpha(frame, shield, x, y)

    cv2.imshow("Mandala Shield", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
