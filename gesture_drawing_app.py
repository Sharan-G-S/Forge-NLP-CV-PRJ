"""
Gesture-Based Drawing Application
Draw on screen using finger tracking
- Index finger up = Drawing mode
- Fist = Erasing mode  
- Peace sign = Change color
- Thumb up = Clear screen
- Open palm = Save drawing

Author: Sharan G S
Date: September 23, 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

# Initialize the MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Setup the camera
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Drawing variables
drawing_canvas = np.zeros((hCam, wCam, 3), np.uint8)
draw_color = (255, 0, 0)  # Blue (BGR format)
brush_thickness = 5
eraser_thickness = 50

# Color palette
colors = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
]
current_color_index = 0

# Drawing state
drawing_mode = False
erasing_mode = False
previous_pos = None

# Gesture detection variables
last_gesture_time = time.time()
gesture_cooldown = 1.0

def detect_fingers(lmList):
    """Detect which fingers are up"""
    if not lmList or len(lmList) < 21:
        return []
    
    fingers = []
    
    # Thumb (different logic due to orientation)
    if lmList[4][1] > lmList[3][1]:  # Right hand
        fingers.append(1 if lmList[4][1] > lmList[3][1] else 0)
    else:  # Left hand
        fingers.append(1 if lmList[4][1] < lmList[3][1] else 0)
    
    # Other fingers
    for finger_id in range(1, 5):
        tip_id = finger_id * 4 + 4
        pip_id = finger_id * 4 + 2
        fingers.append(1 if lmList[tip_id][2] < lmList[pip_id][2] else 0)
    
    return fingers

def detect_gesture(fingers):
    """Detect gesture based on finger positions"""
    if not fingers:
        return "unknown"
    
    fingers_up = sum(fingers)
    
    # Specific gestures
    if fingers == [0, 1, 0, 0, 0]:  # Only index finger
        return "draw"
    elif fingers_up == 0:  # Fist
        return "erase"
    elif fingers == [0, 1, 1, 0, 0]:  # Index and middle
        return "peace"
    elif fingers == [1, 0, 0, 0, 0]:  # Only thumb
        return "clear"
    elif fingers_up >= 4:  # Open palm
        return "save"
    else:
        return "neutral"

def get_color_name(color):
    """Get color name for display"""
    color_names = {
        (255, 0, 0): "Blue",
        (0, 255, 0): "Green", 
        (0, 0, 255): "Red",
        (255, 255, 0): "Cyan",
        (255, 0, 255): "Magenta",
        (0, 255, 255): "Yellow",
        (128, 0, 128): "Purple",
        (255, 165, 0): "Orange"
    }
    return color_names.get(color, "Unknown")

def save_drawing():
    """Save the current drawing"""
    timestamp = int(time.time())
    filename = f"/Users/sharan/TEST/gesture_drawing_{timestamp}.png"
    cv2.imwrite(filename, drawing_canvas)
    print(f"Drawing saved as {filename}")
    return filename

print("=== Gesture-Based Drawing Application ===")
print("👆 Index finger = Draw")
print("✊ Fist = Erase")
print("✌️ Peace sign = Change color")
print("👍 Thumbs up = Clear screen")
print("✋ Open palm = Save drawing")
print("❌ Press 'q' to quit")
print("Author: Sharan G S")
print()

# Initialize the MediaPipe Hands
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirror effect
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find the hand landmarks
        results = hands.process(image)

        # Convert the image color back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks (optional, can be toggled off)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract the landmark list
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                if lmList:
                    # Get finger positions
                    fingers = detect_fingers(lmList)
                    current_gesture = detect_gesture(fingers)
                    
                    # Get index finger tip position
                    index_tip = (lmList[8][1], lmList[8][2])
                    
                    current_time = time.time()
                    
                    # Handle gestures
                    if current_gesture == "draw":
                        drawing_mode = True
                        erasing_mode = False
                        
                        # Draw on canvas
                        if previous_pos:
                            cv2.line(drawing_canvas, previous_pos, index_tip, draw_color, brush_thickness)
                        previous_pos = index_tip
                        
                        # Visual feedback
                        cv2.circle(image, index_tip, brush_thickness, draw_color, -1)
                        
                    elif current_gesture == "erase":
                        drawing_mode = False
                        erasing_mode = True
                        
                        # Erase on canvas
                        cv2.circle(drawing_canvas, index_tip, eraser_thickness, (0, 0, 0), -1)
                        previous_pos = index_tip
                        
                        # Visual feedback
                        cv2.circle(image, index_tip, eraser_thickness, (255, 255, 255), 3)
                        
                    elif current_gesture == "peace":
                        if current_time - last_gesture_time > gesture_cooldown:
                            # Change color
                            current_color_index = (current_color_index + 1) % len(colors)
                            draw_color = colors[current_color_index]
                            last_gesture_time = current_time
                            print(f"Color changed to {get_color_name(draw_color)}")
                        
                        drawing_mode = False
                        erasing_mode = False
                        previous_pos = None
                        
                    elif current_gesture == "clear":
                        if current_time - last_gesture_time > gesture_cooldown:
                            # Clear screen
                            drawing_canvas.fill(0)
                            last_gesture_time = current_time
                            print("Canvas cleared")
                        
                        drawing_mode = False
                        erasing_mode = False
                        previous_pos = None
                        
                    elif current_gesture == "save":
                        if current_time - last_gesture_time > gesture_cooldown:
                            # Save drawing
                            filename = save_drawing()
                            last_gesture_time = current_time
                        
                        drawing_mode = False
                        erasing_mode = False
                        previous_pos = None
                        
                    else:
                        # Neutral state
                        drawing_mode = False
                        erasing_mode = False
                        previous_pos = None
                    
                    # Display current mode and gesture
                    mode_text = ""
                    if drawing_mode:
                        mode_text = "DRAWING"
                        mode_color = draw_color
                    elif erasing_mode:
                        mode_text = "ERASING"
                        mode_color = (255, 255, 255)
                    else:
                        mode_text = "NEUTRAL"
                        mode_color = (128, 128, 128)
                    
                    cv2.putText(image, f'Mode: {mode_text}', (50, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
                    
                    cv2.putText(image, f'Gesture: {current_gesture}', (50, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            # No hand detected
            drawing_mode = False
            erasing_mode = False
            previous_pos = None

        # Combine the camera image with the drawing canvas
        combined_image = cv2.addWeighted(image, 0.7, drawing_canvas, 0.3, 0)
        
        # Add title and current color info
        cv2.putText(combined_image, 'Gesture Drawing', (50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_image, 'Sharan G S', (420, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show current color
        color_name = get_color_name(draw_color)
        cv2.putText(combined_image, f'Color: {color_name}', (350, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
        
        # Show brush thickness
        cv2.putText(combined_image, f'Brush: {brush_thickness}px', (350, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Color palette display
        palette_y = hCam - 60
        for i, color in enumerate(colors):
            x_pos = 50 + i * 40
            cv2.rectangle(combined_image, (x_pos, palette_y), (x_pos + 30, palette_y + 30), color, -1)
            if i == current_color_index:
                cv2.rectangle(combined_image, (x_pos - 2, palette_y - 2), (x_pos + 32, palette_y + 32), (255, 255, 255), 2)
        
        cv2.putText(combined_image, 'Colors:', (50, palette_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add gesture guide
        guide_y = 120
        cv2.putText(combined_image, 'Controls:', (450, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined_image, 'Index: Draw', (450, guide_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined_image, 'Fist: Erase', (450, guide_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined_image, 'Peace: Color', (450, guide_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined_image, 'Thumb: Clear', (450, guide_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(combined_image, 'Palm: Save', (450, guide_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display the combined image
        cv2.imshow('Gesture Drawing - Sharan G S', combined_image)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Clear canvas
            drawing_canvas.fill(0)
            print("Canvas cleared via keyboard")
        elif key == ord('s'):
            # Save drawing
            save_drawing()

# Save final drawing automatically
save_drawing()

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

print("Drawing application ended - Sharan G S")