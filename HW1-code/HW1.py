import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

iris = np.genfromtxt("data/iris.txt",delimiter=None) # load the text file
Y = iris[:,-1] # target value is the last column
X = iris[:,0:-1] # features are the other columns
# Note: indexing with ":" indicates all values (in this case, all rows);
# indexing with a value ("0", "1", "-1", etc.) extracts only that one value (here, columns);
# indexing rows/columns with a range ("1:-1") extracts any row/column in that range.


'''
Problem 1: Python & Data Exploration
'''
# (a)
print('Number of features: %s' %(X.shape[1]))
print('Number of data points: %s' %(X.shape[0]))

# (b)
fig, ax = plt.subplots(nrows = 1, ncols = X.shape[1], figsize = (16, 4))
for i in range(X.shape[1]):
    ax[i].hist(X[:,i],bins=20)
    ax[i].set_xlabel('Length of feature %s' %(i+1))
    ax[i].set_ylabel('Frequency')
plt.show()

# (c)
for i in range(X.shape[1]):
    print('Mean of feature %s:, %.3f' %(i+1, np.mean(X[:,i])))

# (d)
for i in range(X.shape[1]):
    print('Variance of feature %s:, %.3f, '
    'Standard Deviation of feature %s:, %.3f'
    %(i+1, np.var(X[:,i]), i+1, np.std(X[:,i])))

# (e) Normalize data
mean_feature = np.mean(X, axis = 0)
std_feature = np.std(X, axis = 0)
X_norm = np.divide((X - np.tile(mean_feature,(X.shape[0],1))),
    np.tile(std_feature,(X.shape[0],1)))
print(np.mean(X_norm,axis=0))

# (d)
fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4))
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(Y), colors, markers):
    ax[0].scatter(X[Y==l,0], X[Y==l,1], c=c, label=l, marker=m)
    ax[0].set_xlabel('Feature 1')
    ax[0].set_ylabel('Feature 2')
    ax[0].legend(loc = 'lower right')
for l, c, m in zip(np.unique(Y), colors, markers):
    ax[1].scatter(X[Y==l,0], X[Y==l,2], c=c, label=l, marker=m)
    ax[1].set_xlabel('Feature 1')
    ax[1].set_ylabel('Feature 3')
    ax[1].legend(loc = 'lower right')
for l, c, m in zip(np.unique(Y), colors, markers):
    ax[2].scatter(X[Y==l,0], X[Y==l,3], c=c, label=l, marker=m)
    ax[2].set_xlabel('Feature 1')
    ax[2].set_ylabel('Feature 4')
    ax[2].legend(loc = 'lower right')
plt.show()


'''
Problem 2: kNN predictions
'''
import mltools as ml
# We'll use some data manipulation routines in the provided class code
# Make sure the "mltools" directory is in a directory on your Python path, e.g.,
# export PYTHONPATH=${PYTHONPATH}:/path/to/parent/dir
# or add it to your path inside Python:
# import sys
# sys.path.append('/path/to/parent/dir/');
# X,Y = ml.shuffleData(X,Y); # shuffle data randomly
# (This is a good idea in case your data are ordered in some pathological way,
# as the Iris data are)
# Xtr,Xte,Ytr,Yte = ml.splitData(X,Y, 0.75); # split data into 75/25 train/test

# (a)
# Use only first two features of X
X_new,Y_new = ml.shuffleData(X[:,[0,1]],Y);
Xtr,Xte,Ytr,Yte = ml.splitData(X_new,Y_new, 0.75);
# Visualize classification boundary for varying values of K = [1,5,10,50]

for K in [1,5,10,50]:
    knn = ml.knn.knnClassify(Xtr, Ytr, K)
    ml.plotClassify2D(knn, Xtr, Ytr )

# (b) Prediction/ error for training set and test set
K=[1,2,5,10,50,100,200];
errTrain = np.zeros(7)
errTest = np.zeros(7)
for i,k in enumerate(K):
    learner = ml.knn.knnClassify(Xtr, Ytr, k)
    Yhat_tr = learner.predict(Xtr)
    Yhat_te = learner.predict(Xte)
    errTrain[i] = (np.sum(Yhat_tr != Ytr))/len(Ytr)
    errTest[i] = (np.sum(Yhat_te != Yte))/len(Yte)
    plt.semilogx(k, errTrain[i], c='r', marker = 'o')
    plt.semilogx(k, errTest[i], c='g', marker = 's')
plt.show()
