import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import math as math
from numpy import asmatrix as arr
from imp import reload

np.random.seed(0)

'''
Problem 1: Decision Trees
'''
# a) Entropy of class variable
ent = 0.6*math.log(10/6.0,2) + 0.4*math.log(10/4.0,2)
print('Entropy of class variable:, %0.4f' %(ent))
# b)
# Information gain for x_1
ent_1_yes = (3.0/6)*math.log(6.0/3,2) + (3.0/6)*math.log(6.0/3,2)
ent_1_no = (3.0/4)*math.log(4.0/3,2) + (1.0/4)*math.log(4.0,2)
gain_1 = (6.0/10)*(ent - ent_1_yes) + (4.0/10)*(ent - ent_1_no)
print('Information gain for feature 1:, %0.4f' %(gain_1))

# Information gain for x_2
ent_2_yes = (5.0/5)*math.log(5.0/5,2)
ent_2_no = (4.0/5)*math.log(5.0/4,2) + (1.0/5)*math.log(5.0,2)
gain_2 = (5.0/10)*(ent - ent_2_yes) + (5.0/10)*(ent - ent_2_no)
print('Information gain for feature 2:, %0.4f' %(gain_2))

# Information gain for x_3
ent_3_yes = (3.0/7)*math.log(7.0/3,2) + (4.0/7)*math.log(7.0/4,2)
ent_3_no = (1.0/3)*math.log(3.0,2) + (2.0/3)*math.log(3.0/2,2)
gain_3 = (7.0/10)*(ent - ent_3_yes) + (3.0/10)*(ent - ent_3_no)
print('Information gain for feature 3:, %0.4f' %(gain_3))

# Information gain for x_4
ent_4_yes = (2.0/7)*math.log(7.0/2,2) + (5.0/7)*math.log(7.0/5,2)
ent_4_no = (2.0/3)*math.log(3.0/2,2) + (1.0/3)*math.log(3.0,2)
gain_4 = (7.0/10)*(ent - ent_4_yes) + (3.0/10)*(ent - ent_4_no)
print('Information gain for feature 4:, %0.4f' %(gain_4))

# Information gain for x_5
ent_5_yes = (1.0/3)*math.log(3.0,2) + (2.0/3)*math.log(3.0/2,2)
ent_5_no = (3.0/7)*math.log(7.0/3,2) + (4.0/7)*math.log(7.0/4,2)
gain_5 = (3.0/10)*(ent - ent_5_yes) + (7.0/10)*(ent - ent_5_no)
print('Information gain for feature 5:, %0.4f' %(gain_5))

# c) Decision tree
# 1st node: x_2 is long? If yes, return class -1. If no, look at info gain
# 2nd node: x_4 has grade? If yes, look at info gain. If no, return class 1
# 3rd node: x_1 know author? If yes, return 1. If no, return -1
'''
Problem 2: Decision Trees on Kaggle
'''

# a) Load training and validation data
X_data = np.genfromtxt("data/X_train.txt",delimiter=None)
Y_data = np.genfromtxt("data/Y_train.txt",delimiter=None)
Xt = X_data[0:10000]
Yt = Y_data[0:10000]
Xv = X_data[20000:100000]
Yv = Y_data[20000:100000]
#Xt,Yt = ml.shuffleData(Xt,Yt) # reorder randomly (important later)
Xt,_ = ml.transforms.rescale(Xt) # works much better on rescaled data
Xv,_ = ml.transforms.rescale(Xv)

# b) Learn a decision tree classifier on the data
dt = ml.dtree.treeClassify(Xt, Yt, maxDepth = 50)
ythat = arr(dt.predict(Xt))
yvhat = arr(dt.predict(Xv))
Yt_true = arr(Yt)
Yv_true = arr(Yv)
print('Training Error:, %0.4f' %(np.mean(ythat.reshape(Yt_true.shape) != Yt_true)))
print('Validation Error:, %0.4f' %(np.mean(yvhat.reshape(Yv_true.shape) != Yv_true)))

# c) Vary maximum depth
mse_tr = []
mse_va = []
for depth in range(16):
    dt = ml.dtree.treeClassify(Xt, Yt, maxDepth = depth)
    ythat = arr(dt.predict(Xt))
    yvhat = arr(dt.predict(Xv))
    Yt_true = arr(Yt)
    Yv_true = arr(Yv)
    mse_tr_temp = np.mean(ythat.reshape(Yt_true.shape) != Yt_true)
    mse_va_temp = np.mean(yvhat.reshape(Yv_true.shape) != Yv_true)
    mse_tr = np.append(mse_tr, mse_tr_temp)
    mse_va = np.append(mse_va, mse_va_temp)

plt.plot(range(16), mse_tr, 'b-', linewidth=2)
plt.plot(range(16), mse_va, 'g-', linewidth=2)
plt.xlabel('Depth')
plt.ylabel('Error')
plt.show()

# d) Use minLeaf to control complexity
mse_tr = []
mse_va = []
for power in range(2,13):
    leaf = 2**power
    dt = ml.dtree.treeClassify(Xt, Yt, maxDepth = 50, minLeaf = leaf)
    ythat = arr(dt.predict(Xt))
    yvhat = arr(dt.predict(Xv))
    Yt_true = arr(Yt)
    Yv_true = arr(Yv)
    mse_tr_temp = np.mean(ythat.reshape(Yt_true.shape) != Yt_true)
    mse_va_temp = np.mean(yvhat.reshape(Yv_true.shape) != Yv_true)
    mse_tr = np.append(mse_tr, mse_tr_temp)
    mse_va = np.append(mse_va, mse_va_temp)

plt.plot(range(2,13), mse_tr, 'b-', linewidth=2)
plt.plot(range(2,13), mse_va, 'g-', linewidth=2)
plt.xlabel('MinLeaf (exponent)')
plt.ylabel('Error')
plt.show()


# f) ROC curve and AUC area
dt = ml.dtree.treeClassify(Xt, Yt, maxDepth = 5) #, minLeaf = 2**4)
fpr, tpr, tnr = dt.roc(Xv, Yv)
plt.plot(fpr, tpr, 'b-', linewidth = 2)
plt.show()
print('AUC Area:, %0.4f' %(dt.auc(Xv, Yv)))

# g) Use best complexity control value to retrain model and test on test set
Xte = np.genfromtxt("data/X_test.txt",delimiter=None)
Xte,_ = ml.transforms.rescale(Xte) # works much better on rescaled data
# Y_te = np.genfromtxt("data/Y_test.txt",delimiter=None)
dt = ml.dtree.treeClassify(Xt, Yt, maxDepth = 5, minLeaf = 2**4)
Ypred = dt.predictSoft( Xte )
# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat_knn200.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred[:,1]) ).T, '%d, %.2f',
        header='ID,Prob1',comments='',delimiter=',');


'''
Problem 3: Random Forests
'''
'''
# a) Learn a bagged ensemble of decision trees on the training data
m = len(Xt)
mp = len(Xv)
nbag = 25
ensemble = [None]*nbag
Yt_true = arr(Yt)
Yv_true = arr(Yv)
predict_t = np.zeros((m, nbag))
predict_v = np.zeros((mp, nbag))
err_t = []
err_v = []
for i in range(nbag):
    Xi, Yi = ml.bootstrapData(Xt, Yt, n_boot=10000)
    ensemble[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth = 15, minLeaf = 4, nFeatures = 5)
    predict_t[:,i] = ensemble[i].predict(Xt)
    predict_v[:,i] = ensemble[i].predict(Xv)
    if (i == 0) or (i == 4) or (i == 9) or (i == 24):
        ythat = arr(np.mean(predict_t[:,0:i+1], axis=1) > 0.5)
        yvhat = arr(np.mean(predict_v[:,0:i+1], axis=1) > 0.5)
        err_temp_t = np.mean(ythat.reshape(Yt_true.shape) != Yt_true)
        err_temp_v = np.mean(yvhat.reshape(Yv_true.shape) != Yv_true)
        err_t = np.append(err_t, err_temp_t)
        err_v = np.append(err_v, err_temp_v)
print(ythat)
plt.plot([1,5,10,25], err_t, 'b-', linewidth=2)
plt.plot([1,5,10,25], err_v, 'g-', linewidth=2)
plt.xlabel('Number of learners')
plt.ylabel('Error (train blue, validation green)')
plt.show()

# b) Build an ensemble using at least 10k training data and make predictions
# on test data, and upload to Kaggle.com
nbag = 1
mte = len(Xte)
predict_te = np.zeros((mte, nbag))
for i in range(nbag):
    Xi, Yi = ml.bootstrapData(Xt, Yt, n_boot=10000)
    ensemble[i] = ml.dtree.treeClassify(Xi, Yi, maxDepth = 15, minLeaf = 4, nFeatures = 7)
    predict_te[:,i] = ensemble[i].predict(Xte)
Ypred = np.mean(predict_te,axis=1) > 0.5
# Now output a file with two columns, a row ID and a confidence in class 1:
np.savetxt('Yhat_knnEns.txt',
np.vstack( (np.arange(len(Ypred)) , Ypred) ).T, '%d, %.2f',
        header='ID,Prob1',comments='',delimiter=',');
'''
