#to prevent inconsistency between 'X' and 'b' keys in QP
import os
from scipy.io import loadmat, savemat

mat_dir = 'datasets/qp'

for fname in os.listdir(mat_dir):
    if fname.endswith('.mat'):
        path = os.path.join(mat_dir, fname)
        data = loadmat(path)
        # Check if key 'b' is present and 'X' is not.
        if 'b' in data and 'X' not in data:
            print(f"Updating {fname}: renaming key 'b' to 'X'")
            data['X'] = data['b']
            del data['b']
            # Save back the modified data
            savemat(path, data)
        else:
            print(f"No update needed for {fname}")
