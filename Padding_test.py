# test the padding and cur-off frequency

import numpy as np
import os
import glob
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import stats

def FT(signal, spacing):
	oversample 	= 0 										# oversample folds; to be experiemented further
	n 			= 2**(int(np.log(signal.size)/np.log(2))+1 + oversample)
	fourier 	= np.fft.fft(signal, n)
	freq 		= np.fft.fftfreq(n, d=spacing)	
	power 		= np.abs(fourier)
	phase 		= np.angle(fourier)

	return [fourier, power, phase, freq]


def gaussian(x, gamma, beta, alpha):
	return 1/gamma * np.exp(-alpha * np.power(x - beta, 2.)) 	



x = (np.arange(201)-100)/10
ccf = gaussian(x, 1, 0, 0.1)
plt.plot(x, ccf, '.')
plt.title('signal')

ft, power, phase, freq = FT(ccf, 0.1)
plt.plot(np.array(ft).real, np.array(ft).imag, '.')
plt.show()


ift = np.fft.ifft(ft)
plt.plot(np.array(ift).real, np.array(ift).imag, '.')
plt.show()
# In [16]: ift.shape                                                                          
# Out[16]: (65536,)
iccf = abs(ift)
plt.plot(iccf, '.') 
plt.show()

ft = np.fft.fft(ccf)
ift = np.fft.ifft(ft)
# In [18]: ift.shape                                                                          
# Out[18]: (201,)