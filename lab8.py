import numpy as np    
import matplotlib.pyplot as plt

a = np.zeros((4,1))
for k in np.arange(1,4):
    a[k] = 0
    
b = np.zeros((4,1))
for k in np.arange(1,4):
    b[k] = (2/(k*np.pi))*(1-np.cos(k*np.pi))

print(a)
print('\n')
print(b)

steps = 1e-3
t = np.arange(0,20+steps,steps)
T = 8
N = [1,3,15,50,150,1500]
y = 0

for h in [1,2]:

    for i in ([1+(h-1)*3,2+(h-1)*3,3+(h-1)*3]):
        
        for k in np.arange(1,N[i-1]+1):
            b = (2/(k*np.pi))*(1-np.cos(k*np.pi))
            x = b*np.sin(k*(2*np.pi/T)*t)
            y = y + x
            
        
        plt.figure(h, figsize=(10,8))
        
        plt.subplot(3,1,i-(h-1)*3)
        plt.plot(t,y)
        plt.grid(True)
        plt.ylabel('N = %i' %N[i-1])
        if i == 1 or i == 4:
            plt.title('Fourier Series')
        if i == 3 or i == 6:
            plt.xlabel('t[s]')
            plt.show()
        y = 0
            