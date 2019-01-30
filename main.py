import lib
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import signal

# -- PARTIE 1 -- Jouer avec les filtres -- #

# gk = lib.wave_kernel(lambd=24, theta = math.pi/3)
# lib.kernel_plot(gk)

# gk = lib.gaussian_kernel(lambd=24, theta = math.pi/3)
# lib.kernel_plot(gk)

# gk = lib.gabor_kernel(lambd=12, theta = math.pi/3, n = 256)
# gk = lib.gabor_kernel(lambd=12, theta = math.pi/3, sl = 1.0, st = 1.0)
# gk = lib.gabor_kernel(lambd=12, theta = math.pi/3, nl = 2.5)
# lib.kernel_plot(gk)

img = mpimg.imread('src/img2.jpg')

gk = lib.gabor_kernel(lambd=24, theta = math.pi/3)
gab = signal.convolve2d(img, gk, boundary='symm', mode='same')
plt.imshow(np.absolute(gab), cmap="gray")

plt.show()