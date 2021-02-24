import numpy as np    
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con

steps = 1000
w = np.arange(1e3,1e6+steps,steps)


R = 1000
L = .010
C = 100e-9

H_mag_dB = 20*np.log10(  (w/(R*C))  /  (np.sqrt((w**4) + (((1/(R*C))**2)-
                                        (2/(L*C)))*(w**2) + (1/(L*C))**2)))
                
H_angle=((np.pi/2)-np.arctan((w/(R*C))/(-(w**2)+(1/(L*C)))))*(180/np.pi)

def H_angle_adj(H_angle):
    for i in range(len(H_angle)):
        if H_angle[i] > 90:
            H_angle[i] = H_angle[i] - 180
    return H_angle

H_angle = H_angle_adj(H_angle)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(2,1,1)
plt.semilogx(w,H_mag_dB)
plt.grid(True)
plt.ylabel('|H(w)|')
plt.title('Part 1 - Task 1 ')

plt.subplot(2,1,2)
plt.semilogx(w,H_angle)
plt.grid(True)
plt.ylabel('/_H(w)')
plt.xlabel('frequency [rad/s]')
plt.show()
##################################################
num = [1/(R*C),0]
denom = [1,1/(R*C),1/(L*C)]

omega,mag,phase = sig.bode((num,denom),w)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(2,1,1)
plt.semilogx(omega,mag)
plt.grid(True)
plt.ylabel('|H(w)|')
plt.title('Part 1  - Task 2')

plt.subplot(2,1,2)
plt.semilogx(omega,phase)
plt.grid(True)
plt.ylabel('|H(w)|')
plt.xlabel('frequency [rad/s]')
####################################################
myFigSize = (12,12)
plt.figure(figsize=myFigSize)

sys = con.TransferFunction(num,denom)
_ = con.bode(sys,omega, dB = True, Hz = True, deg = True, Plot = True)

####################################################
fs = 2*np.pi*50000
steps = 1/fs
t=np.arange(0,0.01+steps,steps)

x = np.cos(2*np.pi*100*t)+np.cos(2*np.pi*3024*t)+np.sin(2*np.pi*50000*t)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(2,1,1)
plt.plot(t,x)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Part 2 - Non-Filtered Signal')

num_z,denom_z = sig.bilinear(num,denom,fs)
y = sig.lfilter(num_z,denom_z,x)

plt.subplot(2,1,2)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('y(t)')
plt.xlabel('t[s]')
plt.title('Filtered Signal')
