import cv2
import mediapipe as mp
import pyautogui
import time

# Get screen dimensions
screen_w, screen_h = pyautogui.size()
center_x, center_y = screen_w // 2, screen_h // 2  # Center as reference

# Initialize camera and FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Time tracking for blink and scroll detection
blink_time = time.time()
scroll_time = time.time()

# Cursor smoothing factor
smooth_factor = 5  # Adjust for responsiveness

# Dead zone to avoid small unintended movements
dead_zone = 0.02  # Adjust if needed

def clamp(value, min_value, max_value):
    """Ensure value stays within screen bounds."""
    return max(min_value, min(value, max_value))

# Initial cursor position
prev_x, prev_y = center_x, center_y

while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    
    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark
        frame_h, frame_w, _ = frame.shape
        
        # Cursor Movement Based on Eye Tracking
        eye_landmark = landmarks[474]  # Using right eye inner landmark
        
        # Get normalized movement from screen center
        dx = eye_landmark.x - 0.5  # Left(-) / Right(+)
        dy = eye_landmark.y - 0.5  # Up(-) / Down(+)
        
        # Apply dead zone to reduce jitter
        if abs(dx) > dead_zone or abs(dy) > dead_zone:
            move_x = int(center_x + dx * screen_w)  # Move relative to center
            move_y = int(center_y + dy * screen_h)
            
            move_x = clamp(move_x, 10, screen_w - 10)
            move_y = clamp(move_y, 10, screen_h - 10)

            pyautogui.moveTo(move_x, move_y, duration=0.05)

            prev_x, prev_y = move_x, move_y  # Update previous position
        
        # Blink Detection for Click  
        left_eye_top = landmarks[145]
        left_eye_bottom = landmarks[159]

        if (left_eye_top.y - left_eye_bottom.y) < 0.0035:  # Adjusted threshold
            if time.time() - blink_time > 0.7:  # Reduced delay
                pyautogui.click()
                blink_time = time.time()

        # Eye Movement for Scrolling  
        upper_eye = landmarks[386]  
        lower_eye = landmarks[374]  
        
        eye_movement = upper_eye.y - lower_eye.y  

        if eye_movement > 0.015 and time.time() - scroll_time > 0.3:
            pyautogui.scroll(15)  # Scroll up (Fix)
            scroll_time = time.time()
        elif eye_movement < -0.015 and time.time() - scroll_time > 0.3:
            pyautogui.scroll(-15)  # Scroll down
            scroll_time = time.time()

    cv2.imshow('Efficient Eye Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()