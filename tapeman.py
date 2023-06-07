import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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

# Plot a square
square_x = [0, PLATE_X, PLATE_X, 0, 0]
square_y = [0, 0, PLATE_Y, PLATE_Y, 0]
plt.plot(square_x, square_y)
plt.scatter(-front_x+PLATE_X, front_y, s=3, c='C0', label='Front') # Flip the x values for plotting only
plt.scatter(back_x, back_y, s=3, c='C1', label='Back')

plt.xlabel('X / mm')
plt.ylabel('Y / mm')
plt.axis("equal")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Target Locations')
plt.pause(5)

# Make lines
f = 0
b = 0
# Reverse
front_x = np.flip(front_x)
front_y = np.flip(front_y)
back_x = np.flip(back_x)
back_y = np.flip(back_y)

for i in range(len(front_x)+len(back_x)):
    if (i // 12) % 2 == 0:
        plt.plot([-front_x[f]+2*CENTER_X, CENTER_X], [front_y[f], CENTER_Y], c='C0', alpha=0.25)
        plt.pause(0.01)
        f += 1
    else:
        plt.plot([back_x[b], CENTER_X], [back_y[b], CENTER_Y], c='C1', alpha=0.25)
        plt.pause(0.01)
        b += 1

plt.show()

