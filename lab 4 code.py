import numpy as np    
import matplotlib.pyplot as plt

steps = 1e-2
t = np.arange(-10,10+steps,steps)

def u(t):
    u = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] < 0:
            u[i] = 0
        else:
            u[i] = 1
    return u

def convolution(func1,func2):
    newfunc1 = len(func1)
    newfunc2 = len(func2)
    func1extend = np.append(func1,np.zeros((1,newfunc2-1)))
    func2extend = np.append(func2,np.zeros((1,newfunc1-1)))
    result = np.zeros(func1extend.shape)
    
    for i in range(newfunc2 + newfunc1 - 2):
        result[i] = 0
        
        for j in range(newfunc1):
            if (i - j + 1 > 0):
                try:
                    result[i] = result[i]+func1extend[j]*func2extend[i-j+1]
                except:
                    print(i,j)
    return result

def h1(t):
    y = np.exp(2*t) * u(1 - t)
    return y

def h2(t):
    y = u(t-2) - u(t-6)
    return y

def h3(t):
    y = np.cos(.5*np.pi*t) * u(t)
    return y

step = u(t)

func1 = h1(t)
func2 = h2(t)
func3 = h3(t)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(t,func1)
plt.grid(True)
plt.ylabel('h1(t)')
plt.title('Task 1 Plots')

plt.subplot(3,1,2)
plt.plot(t,func2)
plt.grid(True)
plt.ylabel('h2(t)')


plt.subplot(3,1,3)
plt.plot(t,func3)
plt.grid(True)
plt.ylabel('h3(t)')
plt.xlabel('t')

#####################################################


textend = np.arange(2*t[0],2*t[len(t)-1]+steps,steps)
    

d = convolution(func1,step)*steps
e = convolution(func2,step)*steps
f = convolution(func3,step)*steps


myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(textend,d)
plt.grid(True)
plt.ylabel('h1(t)*u(t)')
plt.xlabel('t')
plt.title('CONVOLUTION of h1(t) and u(t)')

plt.subplot(3,1,2)
plt.plot(textend,e)
plt.grid(True)
plt.ylabel('h2(t)*u(t)')
plt.xlabel('t')
plt.title('CONVOLUTION of h2(t) and u(t)')

plt.subplot(3,1,3)
plt.plot(textend,f)
plt.grid(True)
plt.ylabel('h3(t)*u(t)')
plt.xlabel('t')
plt.title('CONVOLUTION of h3(t) and u(t)')

########################################################

def f1(t):
    y = .5*(np.exp(2*t)*u(1-t) + np.exp(2)*u(t-1))
    return y

def f2(t):
    y = (t-2)*u(t-2) - (t-6)*u(t-6)
    return y

def f3(t):
    y = (1/(.5*np.pi))*np.sin(.5*np.pi*t)*u(t)
    return y

f1 = f1(textend)
f2 = f2(textend)
f3 = f3(textend)

myFigSize = (12,12)
plt.figure(figsize=myFigSize)

plt.subplot(3,1,1)
plt.plot(textend,f1)
plt.grid(True)
plt.ylabel('h1(t)*u(t)')
plt.title('Hand Calculated f1(t)')

plt.subplot(3,1,2)
plt.plot(textend,f2)
plt.grid(True)
plt.ylabel('h2(t)*u(t)')
plt.title('Hand Calculated f2(t)')

plt.subplot(3,1,3)
plt.plot(textend,f3)
plt.grid(True)
plt.ylabel('h3(t)*u(t)')
plt.title('Hand Calculated f3(t)')


