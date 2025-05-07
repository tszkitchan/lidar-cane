import os
import numpy as np

import matplotlib.pyplot as plt

# Directory containing data files
data_dir = "data"

# Get a list of all files in the data directory
files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Ensure there are files to choose from
if not files:
    raise FileNotFoundError("No files found in the data directory.")

# Randomly select a file
random_file = np.random.choice(files)
file_path = os.path.join(data_dir, random_file)

# Load the data (assuming it's a CSV with two columns: angle and radius)
data = np.loadtxt(file_path, delimiter=',')
angles = data[:, 0]  # First column: angles in degrees
radii = data[:, 1]   # Second column: radii

# Convert angles to radians for polar plotting
angles_rad = np.deg2rad(angles)

# Create a polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.scatter(angles_rad, radii, s=10, c='blue', alpha=0.7)
ax.set_title(f"Polar Visualization of {random_file}", va='bottom')

# Show the plot
plt.show()