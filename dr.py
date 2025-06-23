import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time

 
mandala = Image.open("ab.png").convert("RGBA")
mandala_np = np.array(mandala)

def overlay_image_alpha(bg, fg, x, y):
    """Overlay fg on bg at (x,y) with alpha blending"""
    h, w = fg.shape[:2]
    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg
    
    
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(bg.shape[1], x + w)
    y_end = min(bg.shape[0], y + h)
    
    fg_x_start = x_start - x
    fg_y_start = y_start - y
    fg_x_end = fg_x_start + (x_end - x_start)
    fg_y_end = fg_y_start + (y_end - y_start)
    
    if fg_x_end <= fg_x_start or fg_y_end <= fg_y_start:
        return bg
    
    alpha_fg = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg
    
    for c in range(3):
        bg[y_start:y_end, x_start:x_end, c] = (
            alpha_fg * fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end, c] + 
            alpha_bg * bg[y_start:y_end, x_start:x_end, c]
        )
    return bg

def rotate_image(image, angle):
    """Rotate image by given angle"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return rotated

def add_glow_effect(image, glow_intensity=30):
    """Add glow effect to the mandala"""
     
    h, w = image.shape[:2]
    glow_size = 20
    glow_canvas = np.zeros((h + 2*glow_size, w + 2*glow_size, 4), dtype=np.uint8)
    
    
    glow_canvas[glow_size:glow_size+h, glow_size:glow_size+w] = image
    
    
    for i in range(1, glow_size):
        
        alpha_mask = glow_canvas[:, :, 3]
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(alpha_mask, kernel, iterations=1)
        
       
        glow_color = [0, 255, 255]  # BGR format
        for c in range(3):
            glow_canvas[:, :, c] = np.where(
                (dilated > 0) & (glow_canvas[:, :, 3] == 0),
                glow_color[c],
                glow_canvas[:, :, c]
            )
        
     
        glow_canvas[:, :, 3] = np.where(
            (dilated > 0) & (glow_canvas[:, :, 3] == 0),
            max(10, glow_intensity - i * 2),
            glow_canvas[:, :, 3]
        )
    
    return glow_canvas

def get_shield(mandala_base, scale, rotation_angle=0, add_glow=False):
    """Create shield with scaling, rotation, and optional glow"""
     
    size = int(150 * scale) + 50
    resized = cv2.resize(mandala_base, (size, size), interpolation=cv2.INTER_AREA)
    
    if rotation_angle != 0:
        resized = rotate_image(resized, rotation_angle)
    
    if add_glow:
        resized = add_glow_effect(resized)
    
    return resized

def get_center_and_depth(hand_landmarks, w, h):
    """Get hand center and depth from landmarks"""
     
    cx = int(hand_landmarks[9].x * w)
    cy = int(hand_landmarks[9].y * h)
    cz = hand_landmarks[9].z
    return (cx, cy), cz

def find_nearest_hand_index(centers, depths):
    """Find the index of the nearest hand (smallest depth value)"""
    if not depths:
        return -1
    return np.argmin(depths)

 
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

 
mandala_cv = cv2.cvtColor(np.array(mandala), cv2.COLOR_RGBA2BGRA)
 
rotation_angle = 0
start_time = time.time()

print("Mandala Shield Application Started")
print("Controls:")
print("- 'q': Quit application")
print("- Move hands closer to camera to increase shield size")
print("- Nearest hand gets rotating shield with glow effect")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
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
    
     n
    current_time = time.time()
    rotation_angle = ((current_time - start_time) * 50) % 360  # 50 degrees per second
    
    if len(centers) == 2:
         
        dist = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
        
        if dist < 200:   
            mid = ((centers[0][0] + centers[1][0]) // 2,
                   (centers[0][1] + centers[1][1]) // 2)
            avg_depth = (depths[0] + depths[1]) / 2
            
             
            scale = max(0.5, min(2.0, 1.5 + abs(avg_depth) * 3))
            
            shield = get_shield(mandala_cv, scale, rotation_angle, add_glow=True)
            x = mid[0] - shield.shape[1] // 2
            y = mid[1] - shield.shape[0] // 2
            frame = overlay_image_alpha(frame, shield, x, y)
            
           
            cv2.putText(frame, "Unified Shield", (mid[0] - 80, mid[1] - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:   
            nearest_idx = find_nearest_hand_index(centers, depths)
            
            for i in range(2):
                 
                scale = max(0.3, min(1.8, 1.2 + abs(depths[i]) * 2.5))
                
                 
                is_nearest = (i == nearest_idx)
                angle = rotation_angle if is_nearest else 0
                
                shield = get_shield(mandala_cv, scale, angle, add_glow=is_nearest)
                x = centers[i][0] - shield.shape[1] // 2
                y = centers[i][1] - shield.shape[0] // 2
                frame = overlay_image_alpha(frame, shield, x, y)
                
                
                if is_nearest:
                    cv2.putText(frame, "Nearest Shield", (centers[i][0] - 60, centers[i][1] - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    elif len(centers) == 1:  # Single hand
        
        scale = max(0.4, min(2.0, 1.3 + abs(depths[0]) * 3))
        
        shield = get_shield(mandala_cv, scale, rotation_angle, add_glow=True)
        x = centers[0][0] - shield.shape[1] // 2
        y = centers[0][1] - shield.shape[0] // 2
        frame = overlay_image_alpha(frame, shield, x, y)
        
        cv2.putText(frame, "Active Shield", (centers[0][0] - 60, centers[0][1] - 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully")