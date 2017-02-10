import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import mltools as ml
import math as math
from numpy import asmatrix as arr
from imp import reload

np.random.seed(0)

'''
Problem 1: Basics of Clustering
'''
# a) Load Iris data restricted to the first two features. Observe clusters
iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file
Y = iris[:,-1] # target value is the last column
X = iris[:,0:2] # first two features
plt.plot(iris[:,0], iris[:,1], 'bo', linewidth=2)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()

'''
# b) k-means on data for k = 5 and 20. Try a few different initializations

# random init
[z5, c5, sumd5] = ml.cluster.kmeans(X, 5, init='random', max_iter=100)
[z20, c20, sumd20] = ml.cluster.kmeans(X, 20, init='random', max_iter=100)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ml.plotClassify2D(None, X, z5, axis=ax[0])
ax[0].plot(c5[:,0], c5[:,1], 'r*', linewidth=10)
ml.plotClassify2D(None, X, z20, axis=ax[1])
ax[1].plot(c20[:,0], c20[:,1], 'r*', linewidth=10)
plt.show()
print('Error rate k = 5, random:, %0.4f' %(np.mean(z5.reshape(Y.shape) != Y)))
print('Error rate k = 20, random:, %0.4f' %(np.mean(z20.reshape(Y.shape) != Y)))

# k++ init
[z5, c5, sumd5] = ml.cluster.kmeans(X, 5, init='k++', max_iter=100)
[z20, c20, sumd20] = ml.cluster.kmeans(X, 20, init='k++', max_iter=100)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ml.plotClassify2D(None, X, z5, axis=ax[0])
ax[0].plot(c5[:,0], c5[:,1], 'r*', linewidth=10)
ml.plotClassify2D(None, X, z20, axis=ax[1])
ax[1].plot(c20[:,0], c20[:,1], 'r*', linewidth=10)
plt.show()
print('Error rate k = 5, k++:, %0.4f' %(np.mean(z5.reshape(Y.shape) != Y)))
print('Error rate k = 20, k++:, %0.4f' %(np.mean(z20.reshape(Y.shape) != Y)))

# farthest init
[z5, c5, sumd5] = ml.cluster.kmeans(X, 5, init='farthest', max_iter=100)
[z20, c20, sumd20] = ml.cluster.kmeans(X, 20, init='farthest', max_iter=100)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ml.plotClassify2D(None, X, z5, axis=ax[0])
ax[0].plot(c5[:,0], c5[:,1], 'r*', linewidth=10)
ml.plotClassify2D(None, X, z20, axis=ax[1])
ax[1].plot(c20[:,0], c20[:,1], 'r*', linewidth=10)
plt.show()
print('Error rate k = 5, farthest:, %0.4f' %(np.mean(z5.reshape(Y.shape) != Y)))
print('Error rate k = 20, farthest:, %0.4f' %(np.mean(z20.reshape(Y.shape) != Y)))

# c) Agglomerative clustering with k=5 and k=20 using single linkage then complete linkage

# single linkage
[z5, join5] = ml.cluster.agglomerative(X, 5, method='min', join=None)
[z20, join20] = ml.cluster.agglomerative(X, 20, method='min', join=None)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ml.plotClassify2D(None, X, z5, axis=ax[0])
ml.plotClassify2D(None, X, z20, axis=ax[1])
plt.show()
print('Error rate k = 5, single linkage:, %0.4f' %(np.mean(z5.reshape(Y.shape) != Y)))
print('Error rate k = 20, single linkage:, %0.4f' %(np.mean(z20.reshape(Y.shape) != Y)))

# complete linkage
[z5, join5] = ml.cluster.agglomerative(X, 5, method='max', join=None)
[z20, join20] = ml.cluster.agglomerative(X, 20, method='max', join=None)
fig, ax = plt.subplots(nrows = 1, ncols =2, figsize = (12, 6))
ml.plotClassify2D(None, X, z5, axis=ax[0])
ml.plotClassify2D(None, X, z20, axis=ax[1])
plt.show()
print('Error rate k = 5, complete linkage:, %0.4f' %(np.mean(z5.reshape(Y.shape) != Y)))
print('Error rate k = 20, complete linkage:, %0.4f' %(np.mean(z20.reshape(Y.shape) != Y)))
'''

'''
Problem 2: EigenFaces

'''
X = np.genfromtxt("data/faces.txt", delimiter=None) # load face dataset
'''
plt.figure()
for i in range(10): # choose the first 10 faces
    img = np.reshape(X[i,:],(24,24)) # convert vectorized data point t
    plt.imshow( img.T , cmap="gray") # display image patch; you may have to squint
    plt.show()
'''

# a) Subtract the mean of the face images
X_0 = X - np.mean(X)

# b) Use scipy.linalg.svd to take SVD of the data
U, S, Vh = linalg.svd(X_0, full_matrices = False)
W = np.dot(U, np.diag(S))
'''
# c) For k = 1,...,10 comute the approximation to X_0 given the first K eigendirections
# and use them to compute the mean squared error in the SVD approximation
mse = []
for k in range(10):
    X_0_hat = np.dot(W[:,:k], Vh[:k,:])
    mse = np.append(mse, np.mean((X_0 - X_0_hat)**2))
plt.plot(range(10), mse, 'bo-', linewidth=2)
plt.xlabel('k')
plt.ylabel('MSE')
plt.show()

# d) Display the first three principal directions of the data, by computing
# mu + alpha V[j,:] and mu - alpha V[j,:].
mu = np.mean(X)
fig, ax = plt.subplots(nrows = 1, ncols =3, figsize = (20, 6))
for i in range(3):
    alpha = 2*np.median(np.abs(W[:,i]))
    img = np.reshape(mu + alpha*Vh[i,:],(24,24)) # img = np.reshape(mu + alpha*Vh[i,:],(24,24))
    ax[i].imshow( img.T , cmap="gray") # display image patch; you may have to squint
plt.show()

# e) Choose two faces and reconstruct them using only the first k principal directions,
# for k = 5, 10, 50

# Using 2nd face
fig, ax = plt.subplots(nrows = 1, ncols =3, figsize = (20, 6))
for k in [5, 10, 20]:
    X_0_hat = np.dot(W[1,:k], Vh[:k,:])
    img = np.reshape(X_0_hat,(24,24)) # convert vectorized data point t
    if k == 5:
        ax[0].imshow( img.T , cmap="gray") # display image patch; you may have to squint
    elif k == 10:
        ax[1].imshow( img.T , cmap="gray") # display image patch; you may have to squint
    elif k == 20:
        ax[2].imshow( img.T , cmap="gray") # display image patch; you may have to squint
plt.show()

# Using 3rd face
fig, ax = plt.subplots(nrows = 1, ncols =3, figsize = (20, 6))
for k in [5, 10, 20]:
    X_0_hat = np.dot(W[2,:k], Vh[:k,:])
    img = np.reshape(X_0_hat,(24,24)) # convert vectorized data point t
    if k == 5:
        ax[0].imshow( img.T , cmap="gray") # display image patch; you may have to squint
    elif k == 10:
        ax[1].imshow( img.T , cmap="gray") # display image patch; you may have to squint
    elif k == 20:
        ax[2].imshow( img.T , cmap="gray") # display image patch; you may have to squint
plt.show()
'''

# f) Choose a few faces at random and display them as images with the coordinates
# given by their coefficients on the first two principal components
idx = np.random.uniform(0, len(X), 15) # pick some data at random or otherwise; get list / vector of inte
coord,params = ml.transforms.rescale( W[:,0:2] ) # normalize scale of "W" locations
for i in idx:
    loc = (coord[i,0],coord[i,0]+0.5, coord[i,1],coord[i,1]+0.5) # where to place
    # the image & size
    img = np.reshape( X[i,:], (24,24) )
    plt.imshow( img.T , cmap="gray", extent=loc ) # draw each image
    plt.axis( (-2,2,-2,2) ) # set axis to reasonable visual scale
plt.show()
