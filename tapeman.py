import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# Constants
PLATE_X = 237.62
PLATE_Y = 237.62
CENTER_X = PLATE_X / 2
CENTER_Y = PLATE_Y / 2

# Read the data from the csv file
front_data = np.genfromtxt("front_data.csv", delimiter=",")
front_x = front_data[:, 0]
front_y = -front_data[:, 1] # Flip the y values cause drawing coordinate is upside down

back_data = np.genfromtxt("back_data.csv", delimiter=",")
back_x = back_data[:, 0]
back_y = -back_data[:, 1] # Flip the y values cause drawing coordinate is upside down

# Reverse order
front_x = np.flip(front_x)
front_y = np.flip(front_y)
back_x = np.flip(back_x)
back_y = np.flip(back_y)


# Setup plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

ax2.set_xlabel('Target Number')
ax2.set_ylabel('Distance from Center / mm')
ax2.set_xlim(-10, 190)
ax2.set_ylim(-10, 160)
ax2.set_title('Distance from Center vs. Target Number')

# Plot a square
square_x = [0, PLATE_X, PLATE_X, 0, 0]
square_y = [0, 0, PLATE_Y, PLATE_Y, 0]
ax1.plot(square_x, square_y)
ax1.scatter(-front_x+PLATE_X, front_y, s=3, c='C0', label='Front') # Flip the x values for plotting only
ax1.scatter(back_x, back_y, s=3, c='C1', label='Back')

ax1.set_xlabel('X / mm')
ax1.set_ylabel('Y / mm')
ax1.axis('equal')
ax1.set_title('Target Locations')
# plt.pause(5)


# Get distance of targets from center
front_radii = np.sqrt((front_x - CENTER_X) ** 2 + (front_y - CENTER_Y) ** 2)
back_radii = np.sqrt((back_x - CENTER_X) ** 2 + (back_y - CENTER_Y) ** 2)

merged_radii = []
for i in range(0, len(front_radii), 12):
    merged_radii.extend(front_radii[i:i+12])
    merged_radii.extend(back_radii[i:i+12])


n = np.arange(1, 181)
arg_n = np.where(((n-1) // 12) % 2 == 1, 0, 1)
front_n = n[arg_n == 1]
back_n = n[arg_n == 0]
for i, nn in enumerate(np.flip(front_n)):
    if (nn+11) % 12 == 0:
        ax1.annotate(nn, (front_x[i], -front_y[i]+2*CENTER_Y), fontsize=8, xytext=(front_x[i], -front_y[i]+2*CENTER_Y+5), c='C0', ha='center')
for i, nn in enumerate(np.flip(back_n)):
    if (nn+11) % 12 == 0:
        ax1.annotate(nn, (back_x[i], -back_y[i]+2*CENTER_Y), fontsize=8, xytext=(back_x[i]+PLATE_X-17.5, -back_y[i]+2*CENTER_Y-10), c='C1', ha='center')


f = 0
b = 0
plt.savefig('frames/frame0.png')
ax2.plot(n[:0], merged_radii[:0], c='C2', alpha=0.25)
for i in range(len(front_x)+len(back_x)):
    if (i // 12) % 2 == 0:
        ax1.plot([-front_x[f]+2*CENTER_X, CENTER_X], [front_y[f], CENTER_Y], c='C0', alpha=0.25)
        ax2.scatter(front_n[f], front_radii[f], c='C0', s=5)
        f += 1
    else:
        ax1.plot([back_x[b], CENTER_X], [back_y[b], CENTER_Y], c='C1', alpha=0.25)
        ax2.scatter(back_n[b], back_radii[b], c='C1', s=5)
        b += 1

    if i%12 == 0:
        ax2.annotate(str(i+1), (n[i], merged_radii[i]), xytext=(n[i], merged_radii[i]+0.75), ha='center', va='bottom')

    ax2.lines.pop(0)
    ax2.plot(n[:i+1], merged_radii[:i+1], c='C2', alpha=0.25)
    plt.pause(0.01)
    plt.savefig(f'frames/frame{i+1}.png')

plt.show()

