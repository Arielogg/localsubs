"""
Utility script to test audio input and list available devices.
Run this first to identify the correct microphone device index.
"""

import sounddevice as sd
import numpy as np
import time


def list_devices():
    """List all audio devices."""
    print("\n" + "=" * 70)
    print("ALL AUDIO DEVICES")
    print("=" * 70)
    
    devices = sd.query_devices()
    
    print("\n📥 INPUT DEVICES (Microphones):")
    print("-" * 50)
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " ⭐ DEFAULT" if i == sd.default.device[0] else ""
            print(f"  [{i:2d}] {device['name'][:45]:<45}{default}")
    
    print("\n📤 OUTPUT DEVICES (Speakers):")
    print("-" * 50)
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            default = " ⭐ DEFAULT" if i == sd.default.device[1] else ""
            print(f"  [{i:2d}] {device['name'][:45]:<45}{default}")
    
    print("\n" + "=" * 70)


def test_microphone(device_index=None, duration=5):
    """Test microphone input with visual feedback."""
    device_name = sd.query_devices(device_index)['name'] if device_index else "default"
    
    print(f"\n🎤 Testing microphone: {device_name}")
    print(f"   Recording for {duration} seconds...")
    print("   Speak into the microphone to see audio levels:\n")
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"   ⚠️  Status: {status}")
        
        # Calculate audio level
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        level = np.sqrt(np.mean(audio**2))
        
        # Visual meter
        bar_length = int(level * 500)  # Scale for visibility
        bar = "█" * min(bar_length, 50)
        spaces = " " * (50 - min(bar_length, 50))
        
        # Color based on level
        if bar_length > 30:
            indicator = "🔴"
        elif bar_length > 10:
            indicator = "🟢"
        else:
            indicator = "⚪"
        
        print(f"\r   {indicator} |{bar}{spaces}| {level:.4f}", end="", flush=True)
    
    try:
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=16000,
            callback=audio_callback,
            blocksize=1600  # 100ms blocks
        ):
            time.sleep(duration)
        
        print("\n\n   ✅ Microphone test complete!")
        return True
        
    except Exception as e:
        print(f"\n\n   ❌ Error: {e}")
        return False


def main():
    print("\n" + "🎙️ " * 20)
    print("     AUDIO DEVICE TESTER")
    print("🎙️ " * 20)
    
    list_devices()
    
    print("\nOptions:")
    print("  1. Test default microphone")
    print("  2. Test specific device by index")
    print("  3. Exit")
    
    while True:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            test_microphone()
        elif choice == "2":
            try:
                idx = int(input("Enter device index: ").strip())
                test_microphone(device_index=idx)
            except ValueError:
                print("Invalid index")
        elif choice == "3":
            break
        else:
            print("Invalid choice")
    
    print("\nGoodbye! 👋\n")


if __name__ == "__main__":
    main()
