import sys
import numpy as np
from omnivoice import OmniVoice

print("Loading model...")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")

print("Generating with speed=1.5...")
res = model.generate(text="Dette er en test", language="no", speed=1.5, instruct="middle-aged, male, low pitch")

arr = res[0] if isinstance(res, list) else res
if hasattr(arr, "numpy"):
    arr = arr.numpy()
elif hasattr(arr, "audio"):
    raw = arr.audio
    arr = raw.numpy() if hasattr(raw, "numpy") else np.array(raw)
else:
    arr = np.array(arr)

print('Len:', len(arr))
print('Max vol:', np.max(np.abs(arr)) if len(arr) > 0 else 0)
