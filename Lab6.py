import numpy as np    
import matplotlib.pyplot as plt
import scipy.signal as sig

steps = 1e-5
t = np.arange(0,2+steps,steps)

def u(t):
    u = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            u[i] = 0
        else:
            u[i] = 1
    return u

def y(t):
    y = (.5 -.5*np.exp(-4*t)+np.exp(-6*t))*u(t)
    return y

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(2,1,1)
plt.plot(t,y(t))
plt.grid(True)
plt.ylabel('Y(t)')
plt.title('Hand Calculated Step Response')

#################################################

num = [1,6,12]
denom = [1,10,24]

tout,yout = sig.step((num,denom),T=t)

plt.subplot(2,1,2)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Computer Calculated Step Response')

#################################################

denom=[1,10,24,0]

[R, P, _] = sig.residue(num,denom)

print('R = ', R, '\nP = ', P)

#################################################

steps = 1e-5
t = np.arange(0,4.5+steps,steps)

num = [25250]
denom = [1, 18, 218, 2036, 9085, 25250, 0]

[R, P, _] = sig.residue(num,denom)

print('R = ', R, '\nP = ', P)

y = 0

for i in range(len(R)):
    a = np.real(P[i])
    w = np.imag(P[i])

    mag_k = np.abs(R[i])
    angle_k = np.angle(R[i])
    
    y = y + (mag_k*np.exp(a*t)*np.cos(w*t + angle_k))*u(t)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(2,1,1)
plt.plot(t,y)
plt.grid(True)
plt.ylabel('y(t)')
plt.title('Step Response Using Cosine Method')

##################################################
denom = [1, 18, 218, 2036, 9085, 25250]

tout2,yout2 = sig.step((num,denom),T=t)

plt.subplot(2,1,2)
plt.plot(tout2,yout2)
plt.grid(True)
plt.ylabel('')
plt.title('Step Response Using scipy.signal.step()')
