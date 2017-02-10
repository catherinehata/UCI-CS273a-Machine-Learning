import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import mltools.logistic2 as lc2
from numpy import asmatrix as arr
from imp import reload

np.random.seed(0)


'''
Problem 1: Perceptrons and Logistic Regression
'''
# a
iris = np.genfromtxt("data/iris.txt",delimiter=None)
X, Y = iris[:,0:2], iris[:,-1] # get first two features & target
X,Y = ml.shuffleData(X,Y) # reorder randomly (important later)
X,_ = ml.transforms.rescale(X) # works much better on rescaled data
XA, YA = X[Y<2,:], Y[Y<2] # get class 0 vs 1
XB, YB = X[Y>0,:], Y[Y>0] # get class 1 vs 2
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

plt.scatter(XA[YA == 0,0], XA[YA == 0,1], c='r', marker='s')
plt.scatter(XA[YA == 1,0], XA[YA == 1,1], c='b', marker='x')
plt.scatter(XB[YB == 2,0], XB[YB == 2,1], c='g', marker='o')
plt.show()
# b
reload(lc2) # helpful if you're modifying the code while in iPython
learner=lc2.logisticClassify2(); # create "blank" learner
learner.classes = np.unique(YA) # define class labels using YA or YB
wts = [0.5,1,-0.25]; # TODO: fill in values
learner.theta=wts; # set the learner's parameters
learner.plotBoundary(XA,YA)


# c
yAhat = arr(learner.predict(XA))
yBhat = arr(learner.predict(XB))
YAtemp = arr(YA)
YBtemp = arr(YB)
print('Error rate of A:, %0.4f' %(np.mean(yAhat.reshape(YAtemp.shape) != YAtemp)))
print('Error rate of B:, %0.4f' %(np.mean(yBhat.reshape(YBtemp.shape) != YBtemp)))

# d
X1s = np.linspace(-3,3,100) # densely sample possible x-values
X2s = np.linspace(-10,10,200)
Xs = np.zeros((X1s.shape[0]*X2s.shape[0],2))
k = 0
l1 = X1s.shape[0]
l2 = X2s.shape[0]
for i in range(l1):
    Xs[k*l2:(k+1)*l2-1,0] = X1s[i]
    for j in range(k*l2,(k+1)*l2):
        Xs[j,1] = X2s[j % l2]
    k +=1
Ys = learner.predict(Xs)
ml.plotClassify2D(learner,Xs,Ys)

# e
# dJ(j)/d(theta) = (sigma(x^(j)theta^T) - y^(j))x^(j)

# f

# g

learnerA=lc2.logisticClassify2()
[it, J01, Jsur]= learnerA.train(XA, YA, initStep=1.0, stopTol=1e-4, stopIter=1001, plot=None)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ax[0].plot(range(0,it), J01,c = 'b')
ax[0].plot(range(0,it), Jsur,c = 'g')
ml.plotClassify2D(learnerA,XA, YA, axis = ax[1])
plt.show()


learnerB=lc2.logisticClassify2()
[it, J01, Jsur]= learnerB.train(XB, YB, initStep=1.0, stopTol=1e-4, stopIter=1001, plot=None)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ax[0].plot(range(0,it), J01,c = 'b')
ax[0].plot(range(0,it), Jsur,c = 'g')
ml.plotClassify2D(learnerB,XB, YB, axis = ax[1])
plt.show()
