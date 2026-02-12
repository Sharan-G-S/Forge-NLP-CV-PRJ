"""
Gesture-Based Presentation Controller
Control PowerPoint/Keynote presentations using hand gestures
- Swipe right = Next slide
- Swipe left = Previous slide
- Point up = Start slideshow
- Fist = Stop/Exit slideshow
- Open palm = Pause

Author: Sharan G S
Date: September 23, 2025
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui
import time
from collections import deque

# Initialize the MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Setup the camera
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Gesture detection variables
gesture_history = deque(maxlen=10)
last_gesture_time = time.time()
gesture_cooldown = 1.5  # seconds between gestures

# Slide control
current_slide = 1
total_slides = 20  # Default, can be updated
presentation_active = False

def detect_gesture(lmList):
    """Detect hand gesture based on landmarks"""
    if not lmList or len(lmList) < 21:
        return "unknown"
    
    # Get key landmark positions
    thumb_tip = lmList[4]
    thumb_mcp = lmList[2]
    index_tip = lmList[8]
    index_mcp = lmList[5]
    middle_tip = lmList[12]
    middle_mcp = lmList[9]
    ring_tip = lmList[16]
    ring_mcp = lmList[13]
    pinky_tip = lmList[20]
    pinky_mcp = lmList[17]
    wrist = lmList[0]
    
    # Calculate if fingers are extended
    fingers = []
    
    # Thumb (different logic due to orientation)
    if thumb_tip[1] > thumb_mcp[1]:  # Right hand
        fingers.append(thumb_tip[1] > thumb_mcp[1])
    else:  # Left hand
        fingers.append(thumb_tip[1] < thumb_mcp[1])
    
    # Other fingers
    fingers.append(index_tip[2] < index_mcp[2])  # Index
    fingers.append(middle_tip[2] < middle_mcp[2])  # Middle
    fingers.append(ring_tip[2] < ring_mcp[2])  # Ring
    fingers.append(pinky_tip[2] < pinky_mcp[2])  # Pinky
    
    # Detect specific gestures
    fingers_up = sum(fingers)
    
    # Fist (no fingers up or just thumb)
    if fingers_up <= 1:
        return "fist"
    
    # Open palm (all fingers up)
    elif fingers_up >= 4:
        return "open_palm"
    
    # Point up (only index finger up)
    elif fingers == [False, True, False, False, False]:
        return "point_up"
    
    # Peace sign (index and middle up)
    elif fingers == [False, True, True, False, False]:
        return "peace"
    
    # Three fingers (thumb, index, middle)
    elif fingers_up == 3 and fingers[0] and fingers[1] and fingers[2]:
        return "three"
    
    # Detect swipe gestures based on hand movement
    elif fingers[1]:  # If index finger is up, check for swipe
        return "swipe_ready"
    
    return "unknown"

def detect_swipe(current_pos, history_positions):
    """Detect swipe direction based on position history"""
    if len(history_positions) < 5:
        return None
    
    # Get recent positions
    recent_positions = list(history_positions)[-5:]
    start_x = recent_positions[0][0]
    end_x = recent_positions[-1][0]
    
    # Check for significant horizontal movement
    movement = end_x - start_x
    
    if abs(movement) > 100:  # Minimum movement threshold
        if movement > 0:
            return "swipe_right"
        else:
            return "swipe_left"
    
    return None

def execute_presentation_command(gesture, swipe_direction=None):
    """Execute presentation control commands"""
    global current_slide, presentation_active, last_gesture_time
    
    current_time = time.time()
    if current_time - last_gesture_time < gesture_cooldown:
        return None
    
    command_executed = None
    
    if gesture == "point_up":
        # Start presentation (F5 for PowerPoint, Command+Shift+P for Keynote)
        try:
            pyautogui.hotkey('cmd', 'shift', 'p')  # Keynote
        except:
            pyautogui.press('f5')  # PowerPoint
        presentation_active = True
        command_executed = "Started Presentation"
        
    elif gesture == "fist":
        # Stop presentation (Escape)
        pyautogui.press('escape')
        presentation_active = False
        command_executed = "Stopped Presentation"
        
    elif swipe_direction == "swipe_right" and presentation_active:
        # Next slide
        pyautogui.press('right')
        current_slide = min(current_slide + 1, total_slides)
        command_executed = f"Next Slide ({current_slide})"
        
    elif swipe_direction == "swipe_left" and presentation_active:
        # Previous slide
        pyautogui.press('left')
        current_slide = max(current_slide - 1, 1)
        command_executed = f"Previous Slide ({current_slide})"
        
    elif gesture == "open_palm":
        # Pause/Resume (Space bar)
        pyautogui.press('space')
        command_executed = "Pause/Resume"
        
    elif gesture == "peace":
        # Show overview (Grid view - depends on presentation software)
        pyautogui.press('g')
        command_executed = "Grid View"
        
    elif gesture == "three":
        # Black screen toggle
        pyautogui.press('b')
        command_executed = "Black Screen Toggle"
    
    if command_executed:
        last_gesture_time = current_time
        
    return command_executed

print("=== Gesture-Based Presentation Controller ===")
print("👆 Point up = Start Presentation")
print("✊ Fist = Stop Presentation")
print("👈➡️ Swipe Left/Right = Previous/Next Slide")
print("✋ Open Palm = Pause/Resume")
print("✌️ Peace Sign = Grid View")
print("🤟 Three Fingers = Black Screen")
print("❌ Press 'q' to quit")
print("Author: Sharan G S")
print()

# Position history for swipe detection
position_history = deque(maxlen=20)

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

        # Add title and status
        cv2.putText(image, 'Presentation Control', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, 'Sharan G S', (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show presentation status
        status_color = (0, 255, 0) if presentation_active else (0, 0, 255)
        status_text = "ACTIVE" if presentation_active else "INACTIVE"
        cv2.putText(image, f'Presentation: {status_text}', (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(image, f'Slide: {current_slide}/{total_slides}', (350, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks
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
                    # Get hand center position for swipe detection
                    hand_center = lmList[9]  # Middle finger MCP joint
                    position_history.append((hand_center[1], hand_center[2]))
                    
                    # Detect current gesture
                    current_gesture = detect_gesture(lmList)
                    
                    # Detect swipe if hand is in swipe ready position
                    swipe_direction = None
                    if current_gesture == "swipe_ready":
                        swipe_direction = detect_swipe(hand_center, position_history)
                    
                    # Execute commands
                    command = execute_presentation_command(current_gesture, swipe_direction)
                    
                    # Display current gesture
                    gesture_display = swipe_direction if swipe_direction else current_gesture
                    cv2.putText(image, f'Gesture: {gesture_display}', (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Display last command
                    if command:
                        cv2.putText(image, f'Command: {command}', (50, 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Visual feedback for gestures
                    if current_gesture == "fist":
                        cv2.circle(image, (hand_center[1], hand_center[2]), 50, (0, 0, 255), 3)
                    elif current_gesture == "open_palm":
                        cv2.circle(image, (hand_center[1], hand_center[2]), 50, (0, 255, 0), 3)
                    elif current_gesture == "point_up":
                        cv2.arrowedLine(image, (hand_center[1], hand_center[2]+30), 
                                       (hand_center[1], hand_center[2]-30), (255, 0, 0), 5)
                    elif swipe_direction:
                        arrow_start = (hand_center[1] - 50, hand_center[2])
                        arrow_end = (hand_center[1] + 50, hand_center[2])
                        if swipe_direction == "swipe_left":
                            arrow_start, arrow_end = arrow_end, arrow_start
                        cv2.arrowedLine(image, arrow_start, arrow_end, (255, 255, 0), 5)

        else:
            # No hand detected
            cv2.putText(image, 'Show your hand', (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            position_history.clear()

        # Add gesture guide
        guide_y = 160
        cv2.putText(image, 'Gestures:', (450, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, 'Point Up: Start', (450, guide_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Fist: Stop', (450, guide_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Swipe: Navigate', (450, guide_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Palm: Pause', (450, guide_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display the image
        cv2.imshow('Presentation Control - Sharan G S', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

print("Presentation controller ended - Sharan G S")