import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Define the 8 vertices of a cube, offset to "float" away from origin
offset = np.array([2, 2, 2])
vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1]
]) + offset

# Define the 6 faces using vertex indices
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
    [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
]

# Plot the cube with transparent faces and dashed edges
cube = Poly3DCollection(faces, alpha=0.2, facecolor='cyan', edgecolor='k', linewidths=1, linestyle='--')
ax.add_collection3d(cube)

# Draw axes as arrows from origin
arrow_length = 6
ax.quiver(0, 0, 0, arrow_length, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)
ax.quiver(0, 0, 0, 0, arrow_length+3.5, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)
ax.quiver(0, 0, 0, 0, 0, arrow_length-0.5, color='k', arrow_length_ratio=0.1, linewidth=1.5)

# # Add axis labels at arrow tips
# ax.text(arrow_length + 0.3, 0, 0, 'X', fontsize=12)
# ax.text(0, arrow_length + 0.3, 0, 'Y', fontsize=12)
# ax.text(0, 0, arrow_length + 0.3, 'Z', fontsize=12)

# Hide the default axes
ax.set_axis_off()

# Set axis limits
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 6])
ax.set_zlim([-1, 6])

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Rotate view: higher elev = looking more from above (foreshortens X/Y axes)
ax.view_init(elev=15, azim=-60)

# Show the plot
plt.tight_layout()
plt.savefig('results/cube.png', dpi=450, bbox_inches='tight')
plt.show()
