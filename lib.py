import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math


def gabor_kernel(lambd = 16.0, theta = 0.0, n = 0, sl = 0.7, st = 1.4, nl = 4.0):
	if n <= 0: n = 1+2*int(nl*lambd)
	gl = -0.5/(sl*sl)
	gt = -0.5/(st*st)
	c = math.cos(theta)/lambd
	s = math.sin(theta)/lambd
	x0 = 0.5*(n-1)*(c+s)
	y0 = 0.5*(n-1)*(c-s)
	sc = 1.0/(2*math.pi*sl*st*lambd*lambd)
	gk = np.empty( (n,n), dtype='complex64' )
	for y in range (0,n):
		for x in range (0,n):
			xr = c*x+s*y-x0  # centering, rotation and scaling
			yr = c*y-s*x-y0  # centering, rotation and scaling
			a = 2.0*math.pi*xr  # wave phase
			gk[y,x] = sc*math.exp(gl*xr*xr+gt*yr*yr)*complex(math.cos(a),math.sin(a))
	return gk

def wave_kernel(lambd = 16.0, theta = 0.0, n = 0, sl = 0.7, st = 1.4, nl = 4.0):
	if n <= 0: n = 1+2*int(nl*lambd)
	c = math.cos(theta)/lambd
	s = math.sin(theta)/lambd
	x0 = 0.5*(n-1)*(c+s)
	y0 = 0.5*(n-1)*(c-s)
	sc = 1.0/(2*math.pi*sl*st*lambd*lambd)
	gk = np.empty( (n,n), dtype='complex64' )
	for y in range (0,n):
		for x in range (0,n):
			xr = c*x+s*y-x0  # centering, rotation and scaling
			a = 2.0*math.pi*xr  # wave phase
			gk[y,x] = sc*complex(math.cos(a),math.sin(a))
	return gk


def gaussian_kernel(lambd = 16.0, theta = 0.0, n = 0, sl = 0.7, st = 1.4, nl = 4.0):
	if n <= 0: n = 1+2*int(nl*lambd)
	gl = -0.5/(sl*sl)
	gt = -0.5/(st*st)
	c = math.cos(theta)/lambd
	s = math.sin(theta)/lambd
	x0 = 0.5*(n-1)*(c+s)
	y0 = 0.5*(n-1)*(c-s)
	sc = 1.0/(2*math.pi*sl*st*lambd*lambd)
	gk = np.empty( (n,n), dtype='complex64' )
	for y in range (0,n):
		for x in range (0,n):
			xr = c*x+s*y-x0  # centering, rotation and scaling
			yr = c*y-s*x-y0  # centering, rotation and scaling
			a = 2.0*math.pi*xr  # wave phase
			gk[y,x] = sc*math.exp(gl*xr*xr+gt*yr*yr)
	return gk

def kernel_plot(k):
	kr = (k.view(np.float32).reshape(k.shape + (2,)))[:,:,0]  # extract real (cos) part
	ki = (k.view(np.float32).reshape(k.shape + (2,)))[:,:,1]  # extract imaginary (sin) part
	mpimg.imsave('res/kr.jpg',kr,cmap="gray")
	mpimg.imsave('res/ki.jpg',ki,cmap="gray")
	fig, (re, im) = plt.subplots(1, 2)  # real and imaginary parts
	re.imshow(kr, cmap='gray')
	re.set_title('Real part')
	re.set_axis_off()
	im.imshow(ki, cmap='gray')
	im.set_title('Imaginary part')
	im.set_axis_off()