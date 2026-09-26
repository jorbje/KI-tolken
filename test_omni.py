import sys
import numpy as np
from omnivoice import OmniVoice

print("Loading model...")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")

try:
    print("Testing with instruct...")
    res = model.generate(text="Dette er en test", language="no", instruct="middle-aged, male, low pitch")
    print("Instruct SUCCESS")
except Exception as e:
    print(f"Instruct FAILED: {e}")

try:
    print("Testing without instruct...")
    res = model.generate(text="Dette er en test", language="no")
    print("No instruct SUCCESS")
except Exception as e:
    print(f"No instruct FAILED: {e}")
