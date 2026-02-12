#!/usr/bin/env python3
"""
macOS Mouse Control Permission Test
This script helps you test and verify mouse control permissions on macOS

Author: Sharan G S
Date: September 23, 2025
"""

import pyautogui
import time
import sys

def test_mouse_permissions():
    """Test mouse control permissions step by step"""
    
    print("🔧 macOS Mouse Control Permission Test")
    print("=" * 50)
    
    # Test 1: Get mouse position
    print("\n1️⃣ Testing mouse position detection...")
    try:
        pos = pyautogui.position()
        print(f"✅ Current mouse position: {pos}")
    except Exception as e:
        print(f"❌ Failed to get mouse position: {e}")
        return False
    
    # Test 2: Small mouse movement
    print("\n2️⃣ Testing small mouse movement...")
    try:
        original_pos = pyautogui.position()
        print(f"Original position: {original_pos}")
        
        # Move mouse slightly
        pyautogui.moveRel(10, 10)
        time.sleep(0.5)
        new_pos = pyautogui.position()
        print(f"New position: {new_pos}")
        
        # Move back
        pyautogui.moveTo(original_pos.x, original_pos.y)
        
        if new_pos != original_pos:
            print("✅ Mouse movement is working!")
        else:
            print("❌ Mouse did not move - permissions needed!")
            return False
            
    except Exception as e:
        print(f"❌ Mouse movement failed: {e}")
        return False
    
    # Test 3: Click test
    print("\n3️⃣ Testing mouse click...")
    print("⚠️  This will perform a click in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"Clicking in {i}...")
        time.sleep(1)
    
    try:
        pyautogui.click()
        print("✅ Mouse click executed!")
    except Exception as e:
        print(f"❌ Mouse click failed: {e}")
        return False
    
    return True

def show_permission_instructions():
    """Show instructions for granting permissions"""
    print("\n" + "=" * 60)
    print("🚨 ACCESSIBILITY PERMISSIONS REQUIRED 🚨")
    print("=" * 60)
    print()
    print("To enable mouse control on macOS, please follow these steps:")
    print()
    print("METHOD 1 - System Settings (macOS Ventura 13+):")
    print("1. Open System Settings")
    print("2. Go to Privacy & Security")
    print("3. Click on 'Accessibility'")
    print("4. Click the '+' button")
    print("5. Find and add 'Terminal' (or 'Python' if available)")
    print("6. Make sure it's enabled (toggle switch ON)")
    print()
    print("METHOD 2 - System Preferences (older macOS):")
    print("1. Open System Preferences")
    print("2. Go to Security & Privacy")
    print("3. Click the 'Privacy' tab")
    print("4. Select 'Accessibility' from the left sidebar")
    print("5. Click the lock icon and enter your password")
    print("6. Add 'Terminal' to the list and check the box")
    print()
    print("METHOD 3 - If using VS Code or another editor:")
    print("1. Follow the same steps but add your code editor instead")
    print("2. Examples: 'Visual Studio Code', 'PyCharm', etc.")
    print()
    print("After granting permissions:")
    print("• Close this terminal completely")
    print("• Open a new terminal window")
    print("• Run this test again")
    print()
    print("=" * 60)

def main():
    """Main test function"""
    print("Starting macOS mouse control permission test...")
    print("Author: Sharan G S")
    
    # Configure pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    
    if test_mouse_permissions():
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Mouse control is working properly!")
        print("✅ You can now use the gesture mouse control application!")
    else:
        print("\n❌ TESTS FAILED!")
        print("🔧 Mouse control permissions need to be granted.")
        show_permission_instructions()
        
        # Try to open System Settings
        import subprocess
        try:
            print("\n🔧 Opening System Settings for you...")
            subprocess.run(['open', 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'], check=False)
        except:
            try:
                subprocess.run(['open', '-b', 'com.apple.systempreferences'], check=False)
            except:
                print("Could not open System Settings automatically.")

if __name__ == "__main__":
    main()