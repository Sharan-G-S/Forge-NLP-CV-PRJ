#!/usr/bin/env python3
"""
macOS Brightness Control Test
This script helps test different brightness control methods on macOS

Author: Sharan G S  
Date: September 23, 2025
"""

import subprocess
import time
import platform

def test_applescript_brightness():
    """Test brightness control using AppleScript"""
    print("🔧 Testing AppleScript brightness control...")
    
    try:
        # Method 1: Direct brightness control
        applescript_cmd = '''
        tell application "System Events"
            tell appearance preferences
                set brightness to 0.8
            end tell
        end tell
        '''
        result = subprocess.run(['osascript', '-e', applescript_cmd], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ AppleScript Method 1: SUCCESS")
            time.sleep(2)
            
            # Set back to medium
            applescript_cmd = '''
            tell application "System Events"
                tell appearance preferences
                    set brightness to 0.5
                end tell
            end tell
            '''
            subprocess.run(['osascript', '-e', applescript_cmd], capture_output=True, timeout=5)
            return True
        else:
            print(f"❌ AppleScript Method 1 failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ AppleScript Method 1 error: {e}")
    
    # Method 2: System Events display control
    try:
        applescript_cmd = 'tell application "System Events" to set brightness of display 1 to 0.7'
        result = subprocess.run(['osascript', '-e', applescript_cmd], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ AppleScript Method 2: SUCCESS")
            time.sleep(2)
            
            # Set back
            subprocess.run(['osascript', '-e', 'tell application "System Events" to set brightness of display 1 to 0.5'], 
                         capture_output=True, timeout=5)
            return True
        else:
            print(f"❌ AppleScript Method 2 failed: {result.stderr}")
            
    except Exception as e:
        print(f"❌ AppleScript Method 2 error: {e}")
    
    return False

def test_brightness_command():
    """Test the brightness command line tool"""
    print("\n🔧 Testing 'brightness' command...")
    
    try:
        # Check if brightness command exists
        result = subprocess.run(['which', 'brightness'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 'brightness' command not found")
            print("💡 Install with: brew install brightness")
            return False
        
        # Test brightness control
        subprocess.run(['brightness', '0.8'], check=True, timeout=5)
        print("✅ Brightness command: SUCCESS")
        time.sleep(2)
        
        # Set back
        subprocess.run(['brightness', '0.5'], check=True, timeout=5)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Brightness command failed: {e}")
    except FileNotFoundError:
        print("❌ 'brightness' command not found")
        print("💡 Install with: brew install brightness")
    except Exception as e:
        print(f"❌ Brightness command error: {e}")
    
    return False

def test_keyboard_simulation():
    """Test brightness control using keyboard simulation"""
    print("\n🔧 Testing keyboard brightness simulation...")
    
    try:
        print("📝 Simulating brightness up (F2 key)...")
        # F2 key for brightness up
        subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 145'], 
                      check=True, timeout=5)
        time.sleep(1)
        
        print("📝 Simulating brightness down (F1 key)...")  
        # F1 key for brightness down
        subprocess.run(['osascript', '-e', 'tell application "System Events" to key code 144'], 
                      check=True, timeout=5)
        
        print("✅ Keyboard simulation: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ Keyboard simulation error: {e}")
    
    return False

def show_permission_instructions():
    """Show instructions for granting necessary permissions"""
    print("\n" + "=" * 60)
    print("🚨 ACCESSIBILITY PERMISSIONS MAY BE REQUIRED 🚨")
    print("=" * 60)
    print()
    print("If brightness control isn't working, you may need to grant permissions:")
    print()
    print("1. System Settings > Privacy & Security > Accessibility")
    print("2. Add 'Terminal' and enable it")
    print()
    print("OR for keyboard simulation:")
    print("1. System Settings > Privacy & Security > Input Monitoring") 
    print("2. Add 'Terminal' and enable it")
    print()
    print("3. For some methods, you may also need:")
    print("   System Settings > Privacy & Security > Automation")
    print("   Enable 'System Events' for Terminal")
    print()
    print("=" * 60)

def main():
    """Main test function"""
    print("🔆 macOS Brightness Control Test")
    print("Author: Sharan G S")
    print("=" * 40)
    
    if platform.system() != "Darwin":
        print("❌ This test is for macOS only")
        return
    
    success_methods = []
    
    # Test different methods
    if test_applescript_brightness():
        success_methods.append("AppleScript")
    
    if test_brightness_command():
        success_methods.append("Brightness Command")
    
    if test_keyboard_simulation():
        success_methods.append("Keyboard Simulation")
    
    # Results
    print("\n" + "=" * 40)
    print("📊 TEST RESULTS:")
    print("=" * 40)
    
    if success_methods:
        print("✅ Working methods:")
        for method in success_methods:
            print(f"   • {method}")
        print("\n🎉 Brightness control should work in the gesture app!")
    else:
        print("❌ No brightness control methods worked")
        show_permission_instructions()
        
        # Try to install brightness command
        print("\n🔧 Attempting to install 'brightness' command...")
        try:
            subprocess.run(['brew', 'install', 'brightness'], check=True, timeout=30)
            print("✅ Installed 'brightness' command successfully!")
            print("🔄 Please run the gesture brightness control again")
        except subprocess.CalledProcessError:
            print("❌ Could not install 'brightness' command")
            print("💡 Try manually: brew install brightness")
        except FileNotFoundError:
            print("❌ Homebrew not found")
            print("💡 Install Homebrew first: https://brew.sh")

if __name__ == "__main__":
    main()