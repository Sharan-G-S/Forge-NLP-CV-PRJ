"""
Gesture-Based Screen Brightness Control
Controls screen brightness using hand gestures - distance between thumb and index finger
Similar to volume control but for screen brightness

Author: Sharan G S
Date: September 23, 2025
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
import platform

# Initialize the MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Brightness range (0.1 to 1.0 for macOS)
minBrightness, maxBrightness = 0.1, 1.0
brightnessBar, brightnessPer = 400, 0

# Setup the camera
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

def set_screen_brightness(brightness):
    """Set screen brightness based on the operating system"""
    try:
        if platform.system() == "Darwin":  # macOS
            brightness = max(0.1, min(1.0, brightness))  # Clamp between 0.1 and 1.0
            
            # Method 1: Try using AppleScript (most reliable)
            try:
                brightness_percent = int(brightness * 100)
                applescript_cmd = f'''
                tell application "System Events"
                    tell appearance preferences
                        set brightness to {brightness}
                    end tell
                end tell
                '''
                subprocess.run(['osascript', '-e', applescript_cmd], check=True, capture_output=True)
                print(f"✅ Brightness set to {brightness_percent}% using AppleScript")
                return
            except subprocess.CalledProcessError:
                pass
            
            # Method 2: Try using brightness command if available
            try:
                subprocess.run(["brightness", str(brightness)], check=True, capture_output=True)
                print(f"✅ Brightness set using brightness command")
                return
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
            
            # Method 3: Try using system_profiler and CoreBrightness (advanced)
            try:
                brightness_percent = int(brightness * 100)
                applescript_cmd = f'tell application "System Events" to set brightness of display 1 to {brightness}'
                subprocess.run(['osascript', '-e', applescript_cmd], check=True, capture_output=True)
                print(f"✅ Brightness set using System Events")
                return
            except subprocess.CalledProcessError:
                pass
            
            # Method 4: Try using keyboard brightness controls
            try:
                # Use keyboard events to simulate brightness keys
                if brightness > 0.7:
                    # Bright - simulate F2 key multiple times
                    for _ in range(5):
                        subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 145'], check=False)
                elif brightness < 0.3:
                    # Dim - simulate F1 key multiple times  
                    for _ in range(5):
                        subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 144'], check=False)
                print(f"✅ Brightness adjusted using keyboard simulation")
                return
            except:
                pass
                
            print(f"⚠️ Could not set brightness to {brightness:.1f} - trying alternative methods")
            
        elif platform.system() == "Windows":
            # Windows brightness control (0-100)
            brightness_percent = int(brightness * 100)
            subprocess.run([
                "powershell", "-Command", 
                f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{brightness_percent})"
            ], check=True)
        elif platform.system() == "Linux":
            # Linux brightness control using xrandr
            brightness = max(0.1, min(1.0, brightness))
            subprocess.run(["xrandr", "--output", "eDP-1", "--brightness", str(brightness)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Could not set brightness to {brightness}: {e}")
    except FileNotFoundError as e:
        print(f"⚠️ Brightness control tool not found: {e}")
        print("💡 Install 'brightness' command: brew install brightness")

def get_current_brightness():
    """Get current screen brightness"""
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(["brightness", "-l"], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse brightness value from output
                for line in result.stdout.split('\n'):
                    if 'brightness' in line:
                        return float(line.split()[-1])
        return 0.5  # Default fallback
    except:
        return 0.5

print("=== Gesture-Based Screen Brightness Control ===")
print("👋 Show your hand to the camera")
print("📏 Adjust brightness with thumb-index finger distance")
print("🔆 Close fingers = dim, open fingers = bright")
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

        # Add title and instructions
        cv2.putText(image, 'Brightness Control', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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
                    # Get the coordinates of the thumb tip and index finger tip
                    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                    x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                    # Draw circles on the thumb tip and index finger tip
                    cv2.circle(image, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
                    cv2.circle(image, (x2, y2), 15, (255, 255, 255), cv2.FILLED)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Calculate the distance between the thumb tip and index finger tip
                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    # Show distance for debugging
                    cv2.putText(image, f'Distance: {int(length)}', (50, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Interpolate the brightness based on the length
                    brightness = np.interp(length, [30, 200], [minBrightness, maxBrightness])

                    # Set the system brightness
                    set_screen_brightness(brightness)
                    
                    # Calculate display values
                    brightnessBar = np.interp(length, [30, 200], [400, 150])
                    brightnessPer = np.interp(length, [30, 200], [0, 100])

                    # Draw the brightness bar
                    cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(image, (50, int(brightnessBar)), (85, 400), (0, 255, 255), cv2.FILLED)
                    cv2.putText(image, f'{int(brightnessPer)}%', (60, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(image, 'Brightness', (10, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Add brightness level indicator
                    if brightnessPer < 30:
                        cv2.putText(image, 'DIM', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                    elif brightnessPer > 70:
                        cv2.putText(image, 'BRIGHT', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        cv2.putText(image, 'NORMAL', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # No hand detected
            cv2.putText(image, 'Show your hand', (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image
        cv2.imshow('Brightness Control - Sharan G S', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

print("Brightness control ended - Sharan G S")