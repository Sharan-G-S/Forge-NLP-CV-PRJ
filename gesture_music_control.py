"""
Gesture-Based Music Player Controller
Control music playback using hand gestures
- Thumbs up = Play/Pause
- Swipe right = Next track
- Swipe left = Previous track
- Pinch = Volume control
- Victory sign = Shuffle toggle
- Fist = Stop

Author: Sharan G S  
Date: September 23, 2025
"""

import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess
import time
from collections import deque
import os

# Initialize the MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Setup the camera
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Music control variables
last_gesture_time = time.time()
gesture_cooldown = 1.0  # seconds between gestures
current_volume = 50
is_playing = False
is_shuffled = False
current_track = "No Track"

# Position history for swipe detection
position_history = deque(maxlen=15)

def execute_music_command(command):
    """Execute music control commands based on the system"""
    try:
        if command == "play_pause":
            # macOS: Use AppleScript for Music app
            subprocess.run(['osascript', '-e', 'tell application "Music" to playpause'], check=True)
        elif command == "next":
            subprocess.run(['osascript', '-e', 'tell application "Music" to next track'], check=True)
        elif command == "previous":
            subprocess.run(['osascript', '-e', 'tell application "Music" to previous track'], check=True)
        elif command == "stop":
            subprocess.run(['osascript', '-e', 'tell application "Music" to stop'], check=True)
        elif command == "shuffle":
            subprocess.run(['osascript', '-e', 'tell application "Music" to set shuffle enabled to not shuffle enabled'], check=True)
        elif command.startswith("volume_"):
            volume = int(command.split("_")[1])
            subprocess.run(['osascript', '-e', f'tell application "Music" to set sound volume to {volume}'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Music command failed: {e}")
        # Fallback to system media keys
        try:
            if command == "play_pause":
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 49'], check=True)
            elif command == "next":
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 124'], check=True) 
            elif command == "previous":
                subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 123'], check=True)
        except:
            print("Could not control media")

def get_current_track_info():
    """Get current playing track information"""
    try:
        result = subprocess.run(['osascript', '-e', 'tell application "Music" to return name of current track'], 
                              capture_output=True, text=True, timeout=1)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except:
        pass
    return "No Track"

def detect_gesture(lmList):
    """Detect hand gesture based on landmarks"""
    if not lmList or len(lmList) < 21:
        return "unknown", None
    
    # Get key landmark positions
    thumb_tip = lmList[4]
    thumb_ip = lmList[3]
    index_tip = lmList[8]
    index_mcp = lmList[5]
    middle_tip = lmList[12]
    middle_mcp = lmList[9]
    ring_tip = lmList[16]
    ring_mcp = lmList[13]
    pinky_tip = lmList[20]
    pinky_mcp = lmList[17]
    
    # Calculate if fingers are extended
    fingers = []
    
    # Thumb (different logic due to orientation)
    fingers.append(thumb_tip[1] > thumb_ip[1])  # Simplified thumb detection
    
    # Other fingers
    fingers.append(index_tip[2] < index_mcp[2])  # Index
    fingers.append(middle_tip[2] < middle_mcp[2])  # Middle
    fingers.append(ring_tip[2] < ring_mcp[2])  # Ring
    fingers.append(pinky_tip[2] < pinky_mcp[2])  # Pinky
    
    fingers_up = sum(fingers)
    
    # Gesture detection
    if fingers_up == 0:
        return "fist", None
    elif fingers == [True, False, False, False, False]:
        return "thumbs_up", None
    elif fingers == [False, True, True, False, False]:
        return "victory", None
    elif fingers_up >= 4:
        return "open_palm", None
    elif fingers[1]:  # Index finger up - check for pinch or swipe
        # Check for pinch (thumb and index close)
        thumb_index_distance = math.hypot(thumb_tip[1] - index_tip[1], thumb_tip[2] - index_tip[2])
        if thumb_index_distance < 40:
            return "pinch", thumb_index_distance
        else:
            return "swipe_ready", None
    
    return "unknown", None

def detect_swipe(current_pos, history_positions):
    """Detect swipe direction based on position history"""
    if len(history_positions) < 8:
        return None
    
    # Get recent positions
    recent_positions = list(history_positions)[-8:]
    start_x = recent_positions[0][0]
    end_x = recent_positions[-1][0]
    
    # Check for significant horizontal movement
    movement = end_x - start_x
    
    if abs(movement) > 120:  # Minimum movement threshold
        if movement > 0:
            return "swipe_right"
        else:
            return "swipe_left"
    
    return None

print("=== Gesture-Based Music Player Controller ===")
print("👍 Thumbs Up = Play/Pause")
print("👈➡️ Swipe Left/Right = Previous/Next Track")
print("👌 Pinch = Volume Control") 
print("✌️ Victory Sign = Shuffle Toggle")
print("✊ Fist = Stop")
print("✋ Open Palm = Current Info")
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

        # Add title and status
        cv2.putText(image, 'Music Controller', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, 'Sharan G S', (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show music status
        status_color = (0, 255, 0) if is_playing else (255, 100, 100)
        status_text = "PLAYING" if is_playing else "PAUSED"
        cv2.putText(image, f'Status: {status_text}', (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show current track (truncated if too long)
        track_display = current_track if len(current_track) <= 25 else current_track[:22] + "..."
        cv2.putText(image, f'Track: {track_display}', (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show volume
        cv2.putText(image, f'Volume: {current_volume}%', (400, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show shuffle status
        shuffle_text = "ON" if is_shuffled else "OFF"
        cv2.putText(image, f'Shuffle: {shuffle_text}', (400, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
                    current_gesture, gesture_value = detect_gesture(lmList)
                    
                    # Detect swipe if hand is in swipe ready position
                    swipe_direction = None
                    if current_gesture == "swipe_ready":
                        swipe_direction = detect_swipe(hand_center, position_history)
                    
                    # Execute commands based on gestures
                    current_time = time.time()
                    command_executed = None
                    
                    if current_time - last_gesture_time >= gesture_cooldown:
                        if current_gesture == "thumbs_up":
                            execute_music_command("play_pause")
                            is_playing = not is_playing
                            command_executed = "Play/Pause"
                            last_gesture_time = current_time
                            
                        elif current_gesture == "fist":
                            execute_music_command("stop")
                            is_playing = False
                            command_executed = "Stop"
                            last_gesture_time = current_time
                            
                        elif current_gesture == "victory":
                            execute_music_command("shuffle")
                            is_shuffled = not is_shuffled
                            command_executed = "Shuffle Toggle"
                            last_gesture_time = current_time
                            
                        elif swipe_direction == "swipe_right":
                            execute_music_command("next")
                            command_executed = "Next Track"
                            last_gesture_time = current_time
                            
                        elif swipe_direction == "swipe_left":
                            execute_music_command("previous")
                            command_executed = "Previous Track"
                            last_gesture_time = current_time
                            
                        elif current_gesture == "open_palm":
                            current_track = get_current_track_info()
                            command_executed = "Track Info Updated"
                            last_gesture_time = current_time
                    
                    # Handle volume control with pinch (real-time, no cooldown)
                    if current_gesture == "pinch" and gesture_value:
                        # Map pinch distance to volume (inverted - closer = louder)
                        new_volume = int(np.interp(gesture_value, [20, 80], [100, 0]))
                        if abs(new_volume - current_volume) > 5:  # Only update if significant change
                            current_volume = new_volume
                            execute_music_command(f"volume_{current_volume}")
                        command_executed = f"Volume: {current_volume}%"
                    
                    # Display current gesture
                    gesture_display = swipe_direction if swipe_direction else current_gesture
                    cv2.putText(image, f'Gesture: {gesture_display}', (50, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    # Display last command
                    if command_executed:
                        cv2.putText(image, f'Command: {command_executed}', (50, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Visual feedback for gestures
                    center_point = (hand_center[1], hand_center[2])
                    
                    if current_gesture == "thumbs_up":
                        cv2.circle(image, center_point, 50, (0, 255, 0), 3)
                        cv2.putText(image, "PLAY/PAUSE", (center_point[0] - 50, center_point[1] - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    elif current_gesture == "fist":
                        cv2.circle(image, center_point, 50, (0, 0, 255), 3)
                        cv2.putText(image, "STOP", (center_point[0] - 25, center_point[1] - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    elif current_gesture == "victory":
                        cv2.circle(image, center_point, 50, (255, 0, 255), 3)
                        cv2.putText(image, "SHUFFLE", (center_point[0] - 35, center_point[1] - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    elif current_gesture == "pinch":
                        cv2.circle(image, center_point, 30, (255, 255, 0), 3)
                        cv2.putText(image, f"VOL: {current_volume}%", (center_point[0] - 40, center_point[1] - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    elif swipe_direction:
                        arrow_start = (center_point[0] - 50, center_point[1])
                        arrow_end = (center_point[0] + 50, center_point[1])
                        if swipe_direction == "swipe_left":
                            arrow_start, arrow_end = arrow_end, arrow_start
                        cv2.arrowedLine(image, arrow_start, arrow_end, (255, 255, 0), 5)
                        track_text = "PREV" if swipe_direction == "swipe_left" else "NEXT"
                        cv2.putText(image, track_text, (center_point[0] - 25, center_point[1] - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        else:
            # No hand detected
            cv2.putText(image, 'Show your hand', (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            position_history.clear()

        # Add gesture guide
        guide_y = 200
        cv2.putText(image, 'Controls:', (450, guide_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, 'Thumbs Up: Play/Pause', (450, guide_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Swipe: Next/Prev', (450, guide_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Pinch: Volume', (450, guide_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Victory: Shuffle', (450, guide_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, 'Fist: Stop', (450, guide_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Display the image
        cv2.imshow('Music Controller - Sharan G S', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()

print("Music controller ended - Sharan G S")