import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import pandas as pd

df = pd.read_csv('/content/CMAC F24 Week 7 Data - Results Formatted.csv')
data = df.values

def double_pendulum(theta11, theta12, L1=1.0, L2=1.0, xplot=0,yplot=0,length=10):
    """
    Visualize a double pendulum given two angles.

    Args:
    - thetaij: Angle of the j-th joint of the i-th pendulum (in radians).
    - L1: Length of the first pendulum arm (default is 1.0).
    - L2: Length of the second pendulum arm (default is 1.0).
    - xplot: The desired x-output
    - yplot: the desired y-output

    Returns:
    - A plot of the double pendulum.
    """
    # Convert angles from degrees to radians
    theta21 = theta11 + np.pi
    theta22 = theta12 + np.pi
    theta31 = theta11 + np.pi/2
    theta32 = theta12 + np.pi/2
    theta41 = theta11 + 1.5*np.pi
    theta42 = theta12 + 1.5*np.pi

    # First pendulum bob (x11, y11)
    x11 = L1 * np.sin(theta11)
    y11 = -L1 * np.cos(theta11)
    x21 = L1 * np.sin(theta21)
    y21 = -L1 * np.cos(theta21)
    x31 = L1 * np.sin(theta31)
    y31 = -L1 * np.cos(theta31)
    x41 = L1 * np.sin(theta41)
    y41 = -L1 * np.cos(theta41)

    # Second pendulum bob (x2, y2)
    x12 = x11 + L2 * np.sin(theta12)
    y12 = y11 - L2 * np.cos(theta12)
    x22 = x21 + L2 * np.sin(theta22)
    y22 = y21 - L2 * np.cos(theta22)
    x32 = x31 + L2 * np.sin(theta32)
    y32 = y31 - L2 * np.cos(theta32)
    x42 = x41 + L2 * np.sin(theta42)
    y42 = y41 - L2 * np.cos(theta42)

    # Create figure
    fig, ax = plt.subplots(figsize=(5 + length, 5))
    width = L1 + L2 # This sets the width and height of the graph so that
                    # even if both legs point in the same direciton, they can still be seen
    ax.set_xlim([-width, width + length])
    ax.set_ylim([-width, width])
    ax.set_aspect('equal')
    ax.set_title('Leg')

    # Plot the first pendulum arms
    ax.plot([0, x11], [0, y11], color='b', lw=2)  # First arm
    ax.plot([x11, x12], [y11, y12], color='r', lw=2)  # Second arm

    # Plot the first pendulum bobs
    ax.plot(x11, y11, 'bo', markersize=10)  # First bob
    ax.plot(x12, y12, 'ro', markersize=10)  # Second bob

    # Plot the second pendulum arms
    ax.plot([0, x21], [0, y21], color='g', lw=2)  # First arm
    ax.plot([x21, x22], [y21, y22], color='k', lw=2)  # Second arm

    # Plot the second pendulum bobs
    ax.plot(x21, y21, 'go', markersize=10)  # First bob
    ax.plot(x22, y22, 'ko', markersize=10)  # Second bob

    # Plot the third pendulum arms
    ax.plot([0 + length, x31 + length], [0, y31], color='b', lw=2)  # First arm
    ax.plot([x31 + length, x32 + length], [y31, y32], color='r', lw=2)  # Second arm

    # Plot the third  pendulum bobs
    ax.plot(x31 + length, y31, 'bo', markersize=10)  # First bob
    ax.plot(x32 + length, y32, 'ro', markersize=10)  # Second bob

    # Plot the fourth pendulum arms
    ax.plot([0 + length, x41 + length], [0, y41], color='g', lw=2)  # First arm
    ax.plot([x41 + length, x42 + length], [y41, y42], color='k', lw=3)  # Second arm

    # Plot the fourth pendulum bobs
    ax.plot(x41 + length, y41, 'go', markersize=10)  # First bob
    ax.plot(x42 + length, y42, 'ko', markersize=10)  # Second bob

    # Plot the body
    ax.plot([0, length], [0, 0], color='k', lw=5)

    ax.plot(xplot,yplot, 'gx', markersize=5)

    # Show plot
    plt.grid(True)
    plt.show()

for datum in data:
  double_pendulum(datum[0],datum[1],datum[2],datum[3],datum[4],datum[5], 3)
  plt.close()

# Example usage
#theta1_input = float(input("Enter the hip angle (degrees): "))
#theta2_input = float(input("Enter the knee angle (degrees): "))
#len1 = float(input("Enter the first length (meters): "))
#len2 = float(input("Enter the second length (meters): "))
#x = float(input("Enter the x (meters): "))
#y = float(input("Enter the y (meters): "))
x = 1
y = 1
theta1_input = 30
theta2_input = 60
len1 = 1
len2 = 2

double_pendulum(theta1_input, theta2_input, len1, len2, x, y)
#double_pendulum(218,117.58,1,1,-0.0677,1.229)


