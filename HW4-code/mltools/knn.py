import numpy as np

from .base import classifier
from .base import regressor
from numpy import asarray as arr
from numpy import asmatrix as mat


################################################################################
## KNNCLASSIFY #################################################################
################################################################################


class knnClassify(classifier):
    """A k-nearest neighbor classifier

    Attributes:
        classes : a list of the possible class labels
        K       :  the number of neighbors to use in the prediction
                alpha   : the (inverse) "bandwidth" for a weighted prediction
                     0 = use unweighted data in the prediction
                     a = weight data point xi proportional to exp( - a * |x-xi|^2 ) 
    """

    def __init__(self, X=None, Y=None, K=1, alpha=0):
        """
        Constructor for knnClassify object.  

        Parameters
        ----------
        X : M x N numpy array 
            M = number of training instances; N = number of features.  
        Y : M x 1 numpy array 
            Contains class labels that correspond to instances in X.
        K : int 
            Sets the number of neighbors to used for predictions.
        alpha : scalar (int or float) 
            Weighted average kernel size (Gaussian kernel; alpha = 0 -> simple average).
        """
        self.K = K
        self.X_train = []
        self.Y_train = []
        self.classes = []
        self.alpha = alpha

        if type(X) == np.ndarray and type(Y) == np.ndarray:
            self.train(X, Y)


    def __repr__(self):
        str_rep = 'knn classifier, {} classes, K={}{}'.format(
            len(self.classes), self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


    def __str__(self):
        str_rep = 'knn classifier, {} classes, K={}{}'.format(
            len(self.classes), self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


## CORE METHODS ################################################################
            

    def train(self, X, Y, K=None, alpha=None):
        """
        This method "trains" the knn classifier: it stores the input data and 
        determines the number of possible classes of data.  Refer to constructor
        doc string for descriptions of X and Y.
        """
        self.X_train = np.asarray(X)
        self.Y_train = np.asarray(Y)
        self.classes = list(np.unique(Y))
        if K is not None:
            self.K = K
        if alpha is not None:
            self.alpha = alpha


    def predictSoft(self, X):
        """
        This method makes a "soft" nearest-neighbor prediction on test data.

        Parameters
        ----------
        X : M x N numpy array 
            M = number of testing instances; N = number of features.  
        """
        mtr,ntr = arr(self.X_train).shape      # get size of training data
        mte,nte = arr(X).shape                 # get size of test data
        if nte != ntr:
            raise ValueError('Training and prediction data must have same number of features')
        
        num_classes = len(self.classes)
        prob = np.zeros((mte,num_classes))     # allocate memory for class probabilities
        K = min(self.K, mtr)                   # (can't use more neighbors than training data points)
        for i in range(mte):                   # for each test example...
            # ...compute sum of squared differences...
            dist = np.sum(np.power(self.X_train - arr(X)[i,:], 2), axis=1)
            # ...find nearest neighbors over training data and keep nearest K data points
            sorted_dist = np.sort(dist, axis=0)[0:K]                
            indices = np.argsort(dist, axis=0)[0:K]             
            wts = np.exp(-self.alpha * sorted_dist)
            count = []
            for c in range(len(self.classes)):
                # total weight of instances of that classes
                count.append(np.sum(wts[self.Y_train[indices] == self.classes[c]]))
            count = np.asarray(count)
            prob[i,:] = np.divide(count, np.sum(count))       # save (soft) results
        return prob

"""
    def predict(self, X):
        ""
        This method makes a nearest neighbor prediction on test data.
        Refer to the predictSoft doc string for a description of X.
        ""
        mtr,ntr = arr(self.X_train).shape      # get size of training data
        mte,nte = arr(X).shape                 # get size of test data
        assert nte == ntr, 'Training and prediction data must have same number of features'
        
        num_classes = len(self.classes)
        Y_te = np.tile(self.Y_train[0], (mte, 1))      # make Y_te same data type as Y_train
        K = min(self.K, mtr)                           # (can't use more neighbors than training data points)
        for i in range(mte):                           # for each test example...
            # ...compute sum of squared differences...
            dist = np.sum(np.power(self.X_train - arr(X)[i,:], 2), axis=1)
            # ...find neares neighbors over training data and keep nearest K data points
            sorted_dist = np.sort(dist, axis=0)[0:K]
            indices = np.argsort(dist, axis=0)[0:K]
            wts = np.exp(-self.alpha * sorted_dist)
            count = []
            for c in range(len(self.classes)):
                # total weight of instances of that classes
                count.append(np.sum(wts[self.Y_train[indices] == self.classes[c]]))
            count = np.asarray(count)
            c_max = np.argmax(count)                   # find largest count...
            Y_te[i] = self.classes[c_max]              # ...and save results
        return Y_te
"""


################################################################################
################################################################################
################################################################################


class knnRegress(regressor):
    """A k-nearest neighbor regressor

    Attributes:
        K       :  the number of neighbors to use in the prediction
                alpha   : the (inverse) "bandwidth" for a weighted prediction
                     0 = use unweighted data in the prediction
                     a = weight data point xi proportional to exp( - a * |x-xi|^2 ) 
    """

    def __init__(self, X=None, Y=None, K=1, alpha=0):
        """
        Constructor for knnRegressor (k-nearest-neighbor regression model).  

        Parameters
        ----------
        X : numpy array
            N x M array of N training instances with M features. 
        Y : numpy array
            1 x N array that contains the values that correspond to instances 
            in X.
        K : int 
            That sets the number of neighbors to used for predictions.
        alpha : scalar 
            Weighted average coefficient (Gaussian weighting; alpha = 0 -> 
            simple average).
        """
        self.K = K
        self.X_train = []
        self.Y_train = []
        self.alpha = alpha

        if X is not None and Y is not None:
            self.train(X, Y)


    def __repr__(self):
        str_rep = 'knnRegress, K={}{}'.format(
            self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


    def __str__(self):
        str_rep = 'knnRegress, K={}{}'.format(
            self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
            if self.alpha else '')
        return str_rep


## CORE METHODS ################################################################
            

    def train(self, X, Y, K=None, alpha=None):
        """
        This method "trains" the knnRegress object: it stores the input data.
        Refer to constructor docstring for descriptions of X and Y.
        """
        self.X_train = np.asarray(X)
        self.Y_train = np.asarray(Y)
        if K is not None:
            self.K = K
        if alpha is not None:
            self.alpha = alpha



    def predict(self, X):
        """
        This method makes a nearest neighbor prediction on test data X.
    
        Parameters
        ----------
        X : numpy array 
            N x M numpy array that contains N data points with M features. 
        """
        ntr,mtr = arr(self.X_train).shape              # get size of training data
        nte,mte = arr(X).shape                         # get size of test data

        if m_tr != m_te:
            raise ValueError('knnRegress.predict: training and prediction data must have the same number of features')

        Y_te = np.tile(self.Y_train[0], (n_te, 1))     # make Y_te the same data type as Y_train
        K = min(self.K, n_tr)                          # can't have more than n_tr neighbors

        for i in range(n_te):
            dist = np.sum(np.power((self.X_train - X[i]), 2), axis=1)  # compute sum of squared differences
            sorted_dist = np.sort(dist, axis=0)[:K]           # find nearest neihbors over X_train and...
            sorted_idx = np.argsort(dist, axis=0)[:K]         # ...keep nearest K data points
            wts = np.exp(-self.alpha * sorted_dist)
            Y_te[i] = arr(wts) * arr(self.Y_train[sorted_idx]).T / np.sum(wts)  # weighted average

        return Y_te





