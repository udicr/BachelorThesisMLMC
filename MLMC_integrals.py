import numpy as np
import time


def f(x):
    return np.cos(np.pi * x[0]) + np.cos(np.pi * x[1])


def fk(x, k):
    return np.cos(k * np.pi * x[0]) + np.cos(k * np.pi * x[1])


'''
Goal is to estimate int_{[0,1]^2} with MC and MLMC Method 
we give eps as spanwidth of the resulting confidence intervall to niveau alpha
or eps as difference to the exact value which might be known
'''


def mc1(d, func, eps=0.005, alpha=0.01, res=None):
    X = []
    fX = []
    mc = 0
    n = 0
    kl = 1
    if res is not None:
        t = time.time()
        while kl > eps:
            x = np.random.uniform(size=d)
            #print(x)
            X.append(x)

            fx = func(x)
            fX.append(fx)

            mc = np.mean(fX)
            kl = np.abs(res - mc)
            #print(kl)
        n = len(fX)
        return (mc, time.time() - t, n, n * d)
    else:
        t = time.time()
        while kl > eps:
            x = np.random.uniform(size=d)
            # print(x)
            X.append(x)

            fx = func(x)
            fX.append(fx)

            mc = np.mean(fX)
            sig = np.var(fX) #Todo . good estimate for Var[X] maybe ststistic?
            kl = 2*2.33* sig/np.sqrt(len(fX))
            print(kl)
        n = len(fX)
        return (mc, time.time() - t, n, n * d)



R, T, F, RG = [], [], [], []
for i in range(100):
    r, t, fv, rg = mc1(2, f,res=0)
    R.append(r)
    T.append(t)
    F.append(fv)
    RG.append(rg)
print("Result : " + str(np.mean(R)))
print("Time : " + str(np.mean(T)))
print("Function Calls : " + str(np.mean(F)))
print("Random generations :" + str(np.mean(RG)))