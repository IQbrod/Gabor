import lib
import math

gk = lib.wave_kernel(lambd=24, theta = math.pi/3)
lib.kernel_plot(gk)