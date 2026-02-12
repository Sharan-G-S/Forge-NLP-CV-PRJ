"""
Gesture-Based Mouse Control
Control mouse cursor and clicking using hand gestures
- Index finger movement controls cursor
- Thumb and index finger together = left click
- Middle finger and index finger = right click

Author: Sharan G S
Date: September 23, 2025
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import pyautogui
import sys
import subprocess

# Configure pyautogui for macOS
pyautogui.FAILSAFE = True  # Move mouse to top-left to stop
pyautogui.PAUSE = 0.01     # Small pause between commands

# Initialize the MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Setup the camera
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Smoothening factor
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Click detection variables
click_threshold = 40
clicking = False
right_clicking = False

def check_accessibility_permissions():
    """Check if accessibility permissions are granted on macOS"""
    try:
        # Test if we can get mouse position
        pyautogui.position()
        # Test if we can move mouse (small movement)
        current_pos = pyautogui.position()
        pyautogui.moveTo(current_pos.x + 1, current_pos.y + 1)
        pyautogui.moveTo(current_pos.x, current_pos.y)
        return True
    except pyautogui.FailSafeException:
        return False
    except Exception as e:
        print(f"Accessibility check failed: {e}")
        return False

def request_accessibility_permissions():
    """Request accessibility permissions on macOS"""
    print("\n" + "="*60)
    print("🚨 ACCESSIBILITY PERMISSIONS REQUIRED 🚨")
    print("="*60)
    print("To control the mouse cursor, please grant accessibility permissions:")
    print()
    print("1. Go to: System Preferences > Security & Privacy > Privacy")
    print("2. Select 'Accessibility' from the left sidebar")
    print("3. Click the lock icon and enter your password")
    print("4. Add 'Terminal' or 'Python' to the list")
    print("5. Make sure the checkbox is checked")
    print()
    print("Alternative method:")
    print("1. Go to: System Settings > Privacy & Security")
    print("2. Click on 'Accessibility'")
    print("3. Add 'Terminal' and enable it")
    print()
    print("After granting permissions, restart this application.")
    print("="*60)
    
    # Try to open System Preferences
    try:
        subprocess.run(['open', '-b', 'com.apple.systempreferences'], check=False)
    except:
        pass
    """Check if a finger is up based on landmarks"""
    if finger_id == 0:  # Thumb
        return lmList[4][1] > lmList[3][1]  # Thumb tip x > thumb IP x
    else:  # Other fingers
        return lmList[finger_id * 4 + 2][2] < lmList[finger_id * 4 + 1][2]  # Tip y < PIP y

def is_finger_up(lmList, finger_id):
    """Check if a finger is up based on landmarks"""
    if finger_id == 0:  # Thumb
        return lmList[4][1] > lmList[3][1]  # Thumb tip x > thumb IP x
    else:  # Other fingers
        return lmList[finger_id * 4 + 2][2] < lmList[finger_id * 4 + 1][2]  # Tip y < PIP y

print("=== Gesture-Based Mouse Control ===")
print("👆 Point with index finger to move cursor")
print("👍 Thumb + Index finger close = Left Click")
print("🤟 Middle + Index finger close = Right Click")
print("✋ All fingers up = No action")
print("❌ Press 'q' to quit")
print("Author: Sharan G S")
print()

# Check accessibility permissions first
if not check_accessibility_permissions():
    request_accessibility_permissions()
    print("\nTesting basic mouse movement...")
    try:
        # Simple test
        current_pos = pyautogui.position()
        print(f"Current mouse position: {current_pos}")
        pyautogui.moveRel(5, 5)
        pyautogui.moveRel(-5, -5)
        print("✅ Mouse control is working!")
    except Exception as e:
        print(f"❌ Mouse control failed: {e}")
        print("Please grant accessibility permissions and try again.")
        sys.exit(1)

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

        # Add title and instructions
        cv2.putText(image, 'Mouse Control', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, 'Sharan G S', (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
                    # Get finger tip coordinates
                    x1, y1 = lmList[8][1], lmList[8][2]  # Index finger tip
                    x2, y2 = lmList[12][1], lmList[12][2]  # Middle finger tip
                    x3, y3 = lmList[4][1], lmList[4][2]   # Thumb tip

                    # Check which fingers are up
                    fingers = []
                    for finger_id in range(5):
                        fingers.append(is_finger_up(lmList, finger_id))

                    # Only index finger up - Moving mode
                    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                        # Convert coordinates to screen coordinates
                        x3_screen = np.interp(x1, (100, wCam - 100), (0, screen_width))
                        y3_screen = np.interp(y1, (100, hCam - 100), (0, screen_height))

                        # Smoothening
                        clocX = plocX + (x3_screen - plocX) / smoothening
                        clocY = plocY + (y3_screen - plocY) / smoothening

                        # Move mouse with error handling
                        try:
                            pyautogui.moveTo(screen_width - clocX, clocY)
                        except Exception as e:
                            print(f"Mouse movement error: {e}")
                            
                        cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                        # Display mode and coordinates for debugging
                        cv2.putText(image, 'MOVING', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, f'X:{int(clocX)} Y:{int(clocY)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

                    # Index and thumb up - Left click mode
                    elif fingers[1] and fingers[0]:
                        # Find distance between index and thumb
                        length = math.hypot(x3 - x1, y3 - y1)
                        
                        # Draw line between thumb and index
                        cv2.line(image, (x1, y1), (x3, y3), (255, 0, 255), 3)
                        cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        cv2.circle(image, (x3, y3), 15, (255, 0, 255), cv2.FILLED)

                        # If fingers are close, perform left click
                        if length < click_threshold:
                            cv2.circle(image, ((x1 + x3) // 2, (y1 + y3) // 2), 15, (0, 255, 0), cv2.FILLED)
                            if not clicking:
                                try:
                                    pyautogui.click()
                                    print("Left click executed")
                                except Exception as e:
                                    print(f"Click error: {e}")
                                clicking = True
                            cv2.putText(image, 'LEFT CLICK', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            clicking = False
                            cv2.putText(image, 'LEFT READY', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # Index and middle finger up - Right click mode
                    elif fingers[1] and fingers[2]:
                        # Find distance between index and middle finger
                        length = math.hypot(x2 - x1, y2 - y1)
                        
                        # Draw line between index and middle finger
                        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        cv2.circle(image, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                        cv2.circle(image, (x2, y2), 15, (0, 255, 255), cv2.FILLED)

                        # If fingers are close, perform right click
                        if length < click_threshold:
                            cv2.circle(image, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 0, 255), cv2.FILLED)
                            if not right_clicking:
                                try:
                                    pyautogui.rightClick()
                                    print("Right click executed")
                                except Exception as e:
                                    print(f"Right click error: {e}")
                                right_clicking = True
                            cv2.putText(image, 'RIGHT CLICK', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            right_clicking = False
                            cv2.putText(image, 'RIGHT READY', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    else:
                        # Reset clicking states
                        clicking = False
                        right_clicking = False
                        cv2.putText(image, 'NEUTRAL', (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    # Show finger states for debugging
                    cv2.putText(image, f'Fingers: {fingers}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            # No hand detected
            cv2.putText(image, 'Show your hand', (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Create a rectangle for the active area
        cv2.rectangle(image, (100, 100), (wCam - 100, hCam - 100), (255, 0, 255), 2)
        cv2.putText(image, 'Active Area', (110, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Display the image
        cv2.imshow('Mouse Control - Sharan G S', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

print("Mouse control ended - Sharan G S")