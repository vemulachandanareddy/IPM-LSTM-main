import os
from scipy.io import loadmat
import numpy as np

mat_dir = 'datasets/qp'
# List of keys you want detailed info on
keys_to_check = ['Q', 'p', 'A', 'X', 'b', 'G', 'c']

for fname in os.listdir(mat_dir):
    if fname.endswith('.mat'):
        path = os.path.join(mat_dir, fname)
        try:
            data = loadmat(path)
            print(f"File: {fname}")
            # Print details for each key of interest
            for key in keys_to_check:
                if key in data:
                    value = data[key]
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: type={type(value)}")
                else:
                    print(f"  {key}: not found")
            # Optionally, print any other keys (ignoring default MATLAB keys)
            other_keys = [k for k in data.keys() if not k.startswith('__') and k not in keys_to_check]
            if other_keys:
                print("  Other keys:", other_keys)
            print("-" * 40)
        except Exception as e:
            print(f"{fname}: Failed to load — {e}")
