import numpy as np    
import matplotlib.pyplot as plt
from scipy.fftpack import fft,fftshift


fs = 300
Ts = 1/fs
t = np.arange(0,2,Ts)

x = np.cos(2*np.pi*t)

def my_fft(x, fs):
    N = len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
              
    return freq, X_mag, X_phi

freq, X_mag, X_phi = my_fft(x,fs)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Task 1 - User-Defined FFT of x(t)')

plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f [Hz]')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('')

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f [Hz]')

############################################
x = 5*np.sin(2*np.pi*t)

freq, X_mag, X_phi = my_fft(x,fs)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Task 2 - User-Defined FFT of x(t)')

plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f [Hz}')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f [Hz]')

###########################################
x = 2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t)+3))**2

freq, X_mag, X_phi = my_fft(x,fs)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('')
plt.xlabel('')
plt.title('Task 3 - User-Defined FFT of x(t)')

plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f[Hz]')

###########################################
x = np.cos(2*np.pi*t)

def my_clean_fft(x, fs):
    N = len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    
    freq = np.arange(-N/2, N/2)*fs/N
    
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for r in range(len(X_phi)):
        if np.abs(X_mag[r]) < 1e-10:
            X_phi[r] = 0
              
    return freq, X_mag, X_phi

freq, X_mag, X_phi = my_clean_fft(x,fs)


myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Task 4 - User-Defined Clean FFT of Task 1 x(t)')

plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f[Hz]')

############################################
x = 5*np.sin(2*np.pi*t)
freq, X_mag, X_phi = my_clean_fft(x,fs)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Task 4 - User-Defined Clean FFT of Task 2 Function')

plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f[Hz]')

###########################################
x = 2*np.cos((2*np.pi*2*t)-2)+(np.sin((2*np.pi*6*t)+3))**2
freq, X_mag, X_phi = my_clean_fft(x,fs)

yFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Task 4 - User-Defined Clean FFT of Task 3 Function')

plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f[Hz]')

###################################################
T = 8
t = np.arange(0,2*T,Ts)
N = 15
y = 0

for k in np.arange(1,N):
            b = (2/(k*np.pi))*(1-np.cos(k*np.pi))
            x = b*np.sin(k*(2*np.pi/T)*t)
            y = y + x

x = y
freq, X_mag, X_phi = my_clean_fft(x,fs)   


yFigSize = (12,12)
plt.figure(figsize=myFigSize)
            
plt.subplot(3,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('x(t)')
plt.xlabel('t[s]')
plt.title('Task 5 - Fourier Series Approximation Clean FFT')



plt.subplot(3,2,3)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)
plt.ylabel('|X(f)|')

plt.subplot(3,2,5)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.ylabel('/_ X(f)')
plt.xlabel('f[Hz]')

plt.subplot(3,2,4)
plt.xlim(-2,2)
plt.stem(freq, X_mag, use_line_collection=True)
plt.grid(True)

plt.subplot(3,2,6)
plt.xlim(-2,2)
plt.stem(freq, X_phi, use_line_collection=True)
plt.grid(True)
plt.xlabel('f[Hz]')


