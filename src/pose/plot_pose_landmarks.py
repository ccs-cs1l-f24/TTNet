import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load JSON data from file
with open("pose_landmarks.json", "r") as json_file:
    pose_data = json.load(json_file)

# Define skeleton connections based on MediaPipe's pose structure
# Each tuple represents a connection between two landmark indices
connections = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Nose to shoulders
    (0, 4), (4, 5), (5, 6), (6, 8),  # Left arm
    (9, 10), (11, 12),  # Torso
    (11, 13), (13, 15),  # Right arm
    (12, 14), (14, 16),  # Left arm
    (11, 23), (12, 24), (23, 24),  # Torso to hips
    (23, 25), (25, 27), (27, 29), (29, 31),  # Right leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # Left leg
]

def plot_landmarks(landmarks, title="Pose Landmarks"):
    # Extract x, y, z coordinates
    x_vals = [landmark["x"] for landmark in landmarks]
    y_vals = [landmark["y"] for landmark in landmarks]
    z_vals = [landmark["z"] for landmark in landmarks]

    # Set up the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_vals, y_vals, z_vals, c='blue', marker='o')

    # Add connecting lines for the skeleton
    for start, end in connections:
        ax.plot(
            [x_vals[start], x_vals[end]],
            [y_vals[start], y_vals[end]],
            [z_vals[start], z_vals[end]],
            'r-',  # Red line for the skeleton
            linewidth=1.5
        )

    # Set axis labels and title
    ax.set_xlabel("Width (X)")
    ax.set_ylabel("Height (Y)")
    ax.set_zlabel("Depth (Z)")
    plt.title(title)
    plt.show()

# Plot landmarks for each blackout mode
for mode in ["left", "right"]:
    print(f"Plotting pose landmarks with {mode} half blacked out")
    for frame_data in pose_data[mode]:
        frame_number = frame_data["frame"]
        landmarks = frame_data["landmarks"]
        plot_landmarks(landmarks, title=f"{mode.capitalize()} Blackout - Frame {frame_number}")
