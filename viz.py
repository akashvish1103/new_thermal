# To vizzualize the PARALLELOPIPED made up from 3 vectors
# NOTE : The VOLUMNE eclosed by this parallelopiped is the DETERMINANT of the 3*3 matrix, where each column is a VECTOR.

import numpy as np
import matplotlib.pyplot as plt

# Three vectors
v1 = np.array([3, 0, 0])
v2 = np.array([1, 2, 0])
v3 = np.array([0, 0, 2])

# Origin
O = np.array([0, 0, 0])

# Vertices
A = v1
B = v2
C = v3

AB = v1 + v2
AC = v1 + v3
BC = v2 + v3
ABC = v1 + v2 + v3

# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# -------------------------------------------------
# Coordinate axes
# -------------------------------------------------

# Axis limits
x_min, x_max = -1, 5
y_min, y_max = -1, 4
z_min, z_max = -1, 4

# X-axis
ax.plot(
    [x_min, x_max],
    [0, 0],
    [0, 0],
    color='black',
    linewidth=1.5
)

# Y-axis
ax.plot(
    [0, 0],
    [y_min, y_max],
    [0, 0],
    color='black',
    linewidth=1.5
)

# Z-axis
ax.plot(
    [0, 0],
    [0, 0],
    [z_min, z_max],
    color='black',
    linewidth=1.5
)

# Axis labels
ax.text(x_max, 0, 0, 'X', fontsize=14)
ax.text(0, y_max, 0, 'Y', fontsize=14)
ax.text(0, 0, z_max, 'Z', fontsize=14)

# -------------------------------------------------
# Draw the three vectors
# -------------------------------------------------

ax.quiver(
    *O, *v1,
    color='red',
    arrow_length_ratio=0.08,
    linewidth=2
)

ax.quiver(
    *O, *v2,
    color='green',
    arrow_length_ratio=0.08,
    linewidth=2
)

ax.quiver(
    *O, *v3,
    color='blue',
    arrow_length_ratio=0.08,
    linewidth=2
)

# Vector labels
ax.text(*v1, '  v1', fontsize=12)
ax.text(*v2, '  v2', fontsize=12)
ax.text(*v3, '  v3', fontsize=12)

# -------------------------------------------------
# Vertices
# -------------------------------------------------

vertices = [
    O, A, B, C,
    AB, AC, BC, ABC
]

# -------------------------------------------------
# Edges of parallelepiped
# -------------------------------------------------

edges = [
    (O, A),
    (O, B),
    (O, C),

    (A, AB),
    (A, AC),

    (B, AB),
    (B, BC),

    (C, AC),
    (C, BC),

    (AB, ABC),
    (AC, ABC),
    (BC, ABC)
]

# Draw edges
for p1, p2 in edges:
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        'k-',
        linewidth=1.2
    )

# Mark vertices
for point in vertices:
    ax.scatter(
        *point,
        color='black',
        s=30
    )

# -------------------------------------------------
# Labels and visualization settings
# -------------------------------------------------

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

ax.set_box_aspect([1, 1, 1])

ax.set_title(
    'Parallelepiped formed by three vectors'
)

plt.show()