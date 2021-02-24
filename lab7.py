import numpy as np    
import matplotlib.pyplot as plt
import scipy.signal as sig


numG = [1,9]
denomG = [1,-2,-40,-64]

numA = [1,4]
denomA = [1,4,3]

p = [1,26,168]

[Z, P, k] = sig.tf2zpk(numG,denomG)
print('Z = ', Z, '\nP = ', P, '\nk = ', k, '\n')

[Z, P, k] = sig.tf2zpk(numA,denomA)
print('Z = ', Z, '\nP = ', P, '\nk = ', k, '\n')

z = np.roots(p)
print('Z = ', z, '\n')

##################################

num = sig.convolve([1,9], [1,4])
denom = sig.convolve([1,-2,-40,-64],[1,4,3])

print('Numerator = ', num)
print('Denominator = ', denom)

tout,yout = sig.step((num,denom))

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(1,1,1)
plt.plot(tout,yout)
plt.grid(True)
plt.ylabel('')
plt.title('open loop convolution of A(s) and G(s)')

######################################

numCL = sig.convolve(numA,numG)
denomCL = sig.convolve(denomG+sig.convolve(p,numG),denomA)

print(numCL)
print(denomCL)

tout1,yout1 = sig.step((numCL,denomCL))

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(1,1,1)
plt.plot(tout1,yout1)
plt.grid(True)
plt.ylabel('')
plt.title('closed loop convolution')

