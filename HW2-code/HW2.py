import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

np.random.seed(0)


'''
Problem 1: Linear Regression
'''
# a
data = np.genfromtxt("data/curve80.txt",delimiter=None) # load the text file
X = data[:,0]
X = X[:,np.newaxis] # code expects shape (M,N) so make sure it's 2-dimensional
Y = data[:,1] # doesn't matter for Y
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.75) # split data set 75/25

# b
lr = ml.linear.linearRegress( Xtr, Ytr ) # create and train model
xs = np.linspace(0,10,200) # densely sample possible x-values
xs = xs[:,np.newaxis] # force "xs" to be an Mx1 matrix (expected by our code)
ys = lr.predict( xs ) # make predictions at xs

print('Linear regression coefficient:')
print(lr.theta)

plt.plot(Xtr, Ytr, 'bo', label='Training data', linewidth=2)
plt.plot(xs, ys, 'g-', label='Prediction', linewidth=2)
plt.xlabel('Scalar feature(s)')
plt.ylabel('Target values')
plt.show()

print('MSE for training data:, %.3f' %lr.mse(Xtr, Ytr))
print('MSE for test data:, %.3f' %lr.mse(Xte, Yte))

# c
fig, ax = plt.subplots(nrows = 1, ncols = 6, figsize = (24, 4))
i = 0
mse_tr = []
mse_te = []
for degree in [1,3,5,7,10,18]:
    params = (None, None)
    # Define a function "Phi(X)" which outputs the expanded and scaled feature matrix:
    Phi = lambda X: ml.transforms.rescale(ml.transforms.fpoly(X, degree,False),
        params)[0]
    # the parameters "degree" and "params" are memorized at the function definition
    # Now, Phi will do the required feature expansion and rescaling:
    lr = ml.linear.linearRegress( Phi(Xtr), Ytr )
    ys_hat = lr.predict( Phi(xs) )
    ax[i].plot(Xtr, Ytr, 'bo')
    ax[i].plot(xs, ys_hat, 'g-', linewidth=2)
    ax[i].set_xlabel('Degree %s' %(degree))
    i+=1
    mse_tr = np.append(mse_tr, lr.mse(Phi(Xtr), Ytr))
    mse_te = np.append(mse_te, lr.mse(Phi(Xte), Yte))
plt.show()

plt.semilogy([1,3,5,7,10,18], mse_tr, 'b-', linewidth=2)
plt.semilogy([1,3,5,7,10,18], mse_te, 'g-', linewidth=2)
plt.xlabel('Degree')
plt.ylabel('MSE')
plt.show()

'''
Problem 2
'''

mse_cv = []
nFolds = 5;
for degree in [1,3,5,7,10,18]:
    params = (None, None)
    # Define a function "Phi(X)" which outputs the expanded and scaled feature matrix:
    Phi = lambda X: ml.transforms.rescale(ml.transforms.fpoly(X, degree,False),
        params)[0]
    # the parameters "degree" and "params" are memorized at the function definition
    J = np.zeros(nFolds)
    for iFold in range(nFolds):
        Xti,Xvi,Yti,Yvi = ml.crossValidate(Xtr,Ytr,nFolds,iFold) # take ith data block as validation
        learner = ml.linear.linearRegress(Phi(Xti), Yti) # train on Xti, Yti , the data for this fold
        J[iFold] = learner.mse(Phi(Xvi), Yvi) # now compute the MSE on Xvi, Yvi and save it
    mse_cv = np.append(mse_cv, np.mean(J))
plt.semilogy([1,3,5,7,10,18], mse_cv, 'b-', linewidth=2)
plt.semilogy([1,3,5,7,10,18], mse_te, 'g-', linewidth=2)
plt.xlabel('Degree')
plt.ylabel('MSE Cross Validation')
plt.show()
