import numpy as np    
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-5
t = np.arange(0,1.2e-3+steps,steps)

def u(t):
    u = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            u[i] = 0
        else:
            u[i] = 1
    return u

R = 1000
L = .027
C = 100e-9

num = [0,1/(R*C),0]
den = [1,1/(R*C),1/(L*C)]

tout,yout = sig.impulse((num,den),T=t)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('')
plt.title('Impulse Response Using sig.impulse()')


def h(t):
    y = (10356*(np.exp(-5000*t))*np.sin(18584*t+(105.1*180/np.pi))*u(t))
    return y

plt.subplot(3,1,2)
plt.plot(t,h(t))
plt.grid(True)
plt.ylabel('')
plt.title('Hand Solved Impulse Response')

tout2,yout2 = sig.step((num,den),T=t)

plt.subplot(3,1,3)
plt.plot(tout2,yout2)
plt.grid(True)
plt.ylabel('')
plt.title('Step Response')

