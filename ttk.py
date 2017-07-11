#!/usr/bin/env python

# ttk.py
# A python implementation of the transductive top k (TTK) model propesed by Liu, et al. (2015)
# in arXiv:1510.05976. The model is a modification of a support vector machine restricted to
# predict at most k examples of a known test set as positive examples (making it a transductive
# method).

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from numpy.linalg import svd
from math import sqrt
import argparse
import logging
import pickle
import datetime

TOLERANCE = 1e-6

# subgradient calculation of the objective w/r/t w and b
# calculated among the training set
def subgrad(w, b, x_train, y_train, C):

    # initialize
    w_grad = w
    b_grad = 0

    for i in range(len(x_train)):
        x_i = x_train[i]
        y_i = y_train[i]

        # add to (w_grad, b_grad) where the hinge loss > 0
        if 1 - y_i * (np.dot(w, x_i) + b) > 0:
            w_grad = w_grad - C * y_i * x_i
            b_grad = b_grad - C * y_i

    return (w_grad, b_grad)


# find the sets of indices of test examples currently predicted
# to be positive (L), on the boundary (E), or negative (R)
# calculated over the test set
def LER(w, b, x_test):

    L = []
    E = []
    R = []

    for j, x_j in enumerate(x_test):

        z = np.dot(w, x_j) + b
        # should this be exactly 0 or 0 within some tolerance (e.g. 10^-10)?
#        if z == 0:
#        if abs(z) <= 0.000000001:
        if abs(z) <= 1e-6:
            E.append(j)
        elif z > 1e-6:
            L.append(j)
        elif z < 1e-6:
            R.append(j)

    return (L, E, R)


# from http://scipy-cookbook.readthedocs.io/items/RankNullspace.html
def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns



# calculate the feasible direction for the descent
# using the subgradient from the training set and test data
# E is list of indices of x_test (predicted on decision bound)
# n_L is number of indices in L (predicted positives)
# k is top k we want precision in
def feasible_dir(w_grad, b_grad, x_test, E, n_L, k):

    x_test_E = sorted([x_test[j] for j in E], key = lambda x : -np.dot(x, w_grad) - b_grad, reverse=True)

    # initialize values
    dw = -w_grad
    db = -b_grad
    B = set()

    # TODO: check if TTK algorithm is expecting 1-indexed data
    # in which case, probably don't want to add 1 here for 0-indexed
#    j0 = min(k - n_L, 1) + 1
    j0 = min(k - n_L, 1)

#    for j_prime in range(j0, len(E)+1):
    for j_prime in range(j0, len(E)):

        any_update = False
#        for j_dbl_prime in range(j_prime, len(E)+1):
        for j_dbl_prime in range(j_prime, len(E)):
            if np.dot(x_test_E[j_dbl_prime], dw) + db > 1e-6:
                any_update = True
                B.add(j_prime)
                # double-check dimensions; might need to be the transpose of this array
                nulls = nullspace(np.array([np.append(x_test_E[jj], 1) for jj in B]))

                # project (dw, db) onto null space of (x_jj, 1) where jj in B

                # https://mail.scipy.org/pipermail/scipy-user/2009-May/021309.html
#                (dw, db) = (nulls * np.dot(nulls, np.append(dw, db))[:,np.newaxis]).sum(axis=0)
                
                # the columns of nulls are an orthonormal set of basis vectors for the nullspace
                # of vectors in B, so the inner dot product gives the distances in each direction
                # and the outer dot product the linear combination of the basis vectors yielding
                # projection:
                proj = np.dot(nulls, np.dot(nulls.T, np.append(dw, db)))
                db = proj[-1]
                dw = proj[:-1]

                break

        if not any_update:
            # presumably this is break and not continue because we've sorted x_test_E?
            break

    return (dw, db)


def calc_objective(w, b, x_train, y_train, C):
    obj = 0.5 * np.dot(w, w)
    for i, x_i in enumerate(x_train):
        obj = obj + C * max(0, 1 - y_train[i]*(np.dot(w, x_i) + b) )
    return obj


def step_length(alpha0, w, b, dw, db, x_train, y_train, C):
    """Calculate the optimal step length for a given iteration of the TTK descent.
    Should be called with 0.5*alpha_min to reduce the risk of hitting the boundary
    of the feasible solution cone. 

    Because of the piecewise-quadratic and convex nature of the objective, we only 
    have to check the set of points where an example crosses the decision boundary
    as well as for a possible minimum between two such points.

    Parameters
    ----------
    alpha0 : float
        Maximum allowable step size to ensure step will yield a feasible solution
    w : ndarray
        Current value of the feature weights vector for the model
    b : float
        Current value of the model bias term
    dw : ndarray
        Feasible descending direction for the w parameter
    db : float
        Fesible descending direction for the b parameter
    x_train : ndarray
        Matrix of the training examples
    y_train : ndarray
        Array of labels for the training examples in [-1, 1]
    C : float
        Slack penalty parameter of the models

    Return value
    ------------
    alpha_star : float
        The optimal step size for the current iteration of the descent
    """

    # number of examples in the training set
    n_ex = len(x_train)

    # the current value of the loss (1 - y*(w'x + b)) for each training example [including negative values, no hinge here]
    lossFar = np.array([1.0 - y_train[i] * (np.dot(x_train[i], w) + b) for i in range(n_ex)])
    # how much the loss will move with a step of (dw, db)
    gradInst0 = np.array([y_train[i] * (np.dot(x_train[i], dw) + db) for i in range(n_ex)])
    # including the C parameter
    gradInst = -C * gradInst0
    # and ignoring examples with lossFar < 0, so just considering those that contribute to loss
    gradInst[lossFar < 0] = 0
    # sum over examples, aggregate change in loss if no examples cross margin
    gradSum = sum(gradInst)

    # for all examples, how big of a step to get to the margin
    lengthToZero = np.array([lossFar[i] / gradInst0[i] if gradInst0[i] != 0 else np.Inf for i in range(n_ex)])
    # flag any examples where a step smaller than alpha0 will get to the margin
    # note that these are examples in the training set where alpha0 was initially determined
    # using the test set such that no test points will cross the decision boundary
    flag = (-1e-16 < lengthToZero) & (lengthToZero < alpha0)

    # identify the indices for examples that would cross the boundary
    points = np.where(flag)[0]

    # if no examples will cross the boundary with a step size of alpha0, we can either use alpha0 as
    # our step size or look for a minimum with a smaller step (step1 below). To calculate that minimum,
    # note that in the region where no examples cross the margin, the objective is quadratic in alpha
    # and a minimum can be found by differentiating w/r/t alpha and setting equal to 0:
    #       0.5 * (w + alpha * dw)'(w + alpha * dw) + C * sum[1 - y * ((w + alpha * dw)'x + b + alpha * db)]
    # (where ' denotes transpose and the sum is taken over all points with 1-y*(w'x + b) > 0)
    if points.size == 0:
        if gradSum < 0:
            step1 = -(gradSum + np.dot(w, dw))/(np.dot(dw, dw))
            return min(alpha0, step1)
        else:
            # if the aggregate gradient isn't decreasing, return no step
            return 0

    # if we do have examples that will cross the margin, the objective is piecewise-quadratic between
    # them and convex overall, so we only need to check step lengths that correspond to each example
    # crossing the margin or look for a minimum in one of these quadratic regions between two such
    # elbow points

    # find the examples that will cross zero with a step smaller than alpha0 and sort them
    # by their length to zero to consider each in turn
    lengthToZero = lengthToZero[flag]
    sind = lengthToZero.argsort()
    lengthCand = lengthToZero[sind]
    cand = points[sind]
    
    # Next we'll iterate through each length candidate and look at the example that's crossing
    # the margin at that step length. We'll need to figure out whether it's moving from
    # contributing to the loss to not or vice-versa and then assess whether the objective is
    # descending, at a minimum, or ascending
    for ind, ix in enumerate(cand):
        # step length at which the example in question crosses the margin
        curLength = lengthCand[ind]

        # this example's contribution to the gradient d(objective) / d(alpha)
        # on the side of the point we're looking at where it contributes to the loss
        gradEll = -C * y_train[ix] * (np.dot(x_train[ix], dw) + db)

        # if gradEll < 0, we must be moving from this example contributing to the loss to
        # it not contributing (so, exiting the margin). Thus, on the left, we have gradEll
        # and on the right we have 0
        if gradEll < 0:
            gradInsLeft = gradEll
            gradInsRight = 0

        # by contrast, if gradEll > 0, we're moving from the example not contributing to the
        # loss to it contributing (coming into the margin). Thus, on the left we have 0 and
        # on the right we have gradEll
        else:
            gradInsRight = gradEll
            gradInsLeft = 0

        # figure out the value of the gradient on the left and right of alpha = curLength
        # gradSum includes all points contributing to the loss up to here (note that it's
        # getting updated as we progress through the for loop).
        # For example, if this point is moving into the margin:
        #       d(obj)/d(alpha)|left = w'dw + (alpha)*(dw)'dw + C * sum[-y[i]*((dw)'x[i] + db)]
        #       d(obj)/d(alpha)|right = w'dw + (alpha)*(dw)'dw + C * sum[-y[i]*((dw)'x[i] + db)] + C * [-y[j]*((dw)'x[j] + db)]
        # where the sum over i is over the points contributing to the loss on the left (gradSum) and 
        # j is the current point (gradEll) and the leading terms come in with gradSecond below.
        # In the case where we're moving an example out of the margin, the left and right gradients will
        # be reversed and we just have to be careful in the calcuation about what's been included in gradSum
        # already...

        # remove this point's contribution from gradSum if necessary and add gradInsLeft to account for
        # which direction we're moving
        gradLinLeft = gradSum - gradInst[ix] + gradInsLeft
        # update gradSum to the state of alpha >= curLength (this will persist to the next iteration)
        gradSum = gradSum - gradInst[ix] + gradInsRight
        # on the right, use the updated value of gradSum
        gradLinRight = gradSum
        # include the leading terms in the gradient, see the comment above
        gradSecond = np.dot(w + curLength*dw, dw)
        gradLeft = gradLinLeft + gradSecond
        gradRight = gradLinRight + gradSecond

        # Figure out whether at the current length we're:
        #   * at a minimum (negative gradient on the left and positive/zero on the right)
        #   * still descending (negative gradients in both directions)
        #   * not descending (positive/zero in both directions)
        #   * not in a convex function (positive/zero on the left and negative on the right)
        if gradLeft < 0 and gradRight >= 0:
            # local minimum found
            return curLength

        elif gradLeft < 0 and gradRight < 0:
            # still descending
            continue

        elif gradLeft >= 0 and gradRight >= 0:
            # not a descending direction
            if ind == 0:
                # if we've hit this at the first point, must not be a step to take
                return 0
            else:
                # look for a minimum to the left of this point (see above for more on the calculation)
                stepLength = -1.0*(gradLinLeft + np.dot(w, dw)) / (np.dot(dw, dw))
                # if that minimum doesn't fall between the previous and current lengths, something has gone wrong
                if not ( (lengthCand[ind - 1] - 1e-12 <= stepLength) and (stepLength <= lengthCand[ind] + 1e-12) ):
                    raise ValueError('Incorrect qudratic step length calculation.')
                else:
                    return stepLength

        else:
            raise ValueError('Not Convex')

    # finally, if we get all the way through the for loop without returning, look for a minimum to the right
    # of the final point and return the smaller of that length and alpha0
    stepLength = -1.0*(gradLinRight + np.dot(w, dw)) / (np.dot(dw, dw))

    return min(stepLength, alpha0)


def run_ttk(x_train, y_train, x_test, k, C=1.0, init_w=None, init_b=0, MAX_ITER=5000):
    """Main method for running the transductive top k model.

    Parameters
    ----------
    x_train : ndarray
        Matrix of the training examples
    y_train : ndarray
        Array of labels for the training examples in [-1, 1]
    x_test : ndarray
        Matrix of the test examples
    k : int
        Number (at most) of test examples to predict as positives
    C : float
        Slack penalty parameter of the models
    init_w : ndarray
        Optional parameter for the starting point for the TTK model
    init_b : float
        Optional parameter for the starting point for the TTK model
    MAX_ITER : int
        Maximum number of iterations to run in fitting the model

    If init_w and init_b are not specified, the model will initialize with a Linear SVM
    model with b adjusted such that k test examples will fall above the decision boundary

    Return value
    ------------
    w : ndarray
        Array of feature weights from the model
    b : float
        Model bias term
    """

    # Either use the initial w & b parameter passed into the method or initial by running
    # a linear SVM and adjusting b such that k examples are above the boundary
    if init_w is None:
        clf = LinearSVC(C=C)
        clf.fit(x_train, y_train)
        w = clf.coef_[0]           # primal problem coef
        b = clf.intercept_[0]

        # score the test examples and shift b so the kth-highest score is 0
        # (hence, k examples would be predicted in the positive class)
        svm_score = sorted(clf.decision_function(x_test), reverse=True)
        b = b - svm_score[k]    # index k will give k+1 element, so k will have score > 0
    
    else:
        # use the user-specified initial conditions
        w = init_w
        b = init_b
        init_score = sorted([np.dot(x_i, w) + b for x_i in x_test], reverse=True)
        b = b - init_score[k]
    

    logging.info('INITIAL OBJECTIVE: {}'.format(calc_objective(w, b, x_train, y_train, C)))

    # run the subgradient descent with at most MAX_ITER iterations
    for i in range(MAX_ITER):

        # calculate the subgradient for the current model and project it onto
        # the null space of a subset of the test set instances on the decision
        # boundary (E), yielding dw and db -- see the original paper for details
        w_grad, b_grad = subgrad(w, b, x_train, y_train, C)
        L, E, R = LER(w, b, x_test)
        dw, db = feasible_dir(w_grad, b_grad, x_test, E, len(L), k)

        # alpha_min here is the minimum value of the step descending step size that will
        # result in a point in R (the currently predicted negative examples) crossing the
        # decision boundary, which is the maximum step size we could take to guarantee
        # feasible descending step without violating the constraint of no more than k
        # positive examples (admittedly, alpha_min isn't a great name since it's the
        # largest-allowable step size)
        alpha_min = None
        for j in R:
            if np.dot(x_test[j], dw) + db > 1e-6:
                alpha = -1 * (np.dot(x_test[j], w) + b) / (np.dot(x_test[j], dw) + db)
                if alpha_min is None or alpha < alpha_min:
                    alpha_min = alpha

        # if alpha_min is still None at this point, no points in R will move towards the
        # decision boundary, so let's just set alpha_min=1.0 in that case?
        if alpha_min is None:
            alpha_min = 1.0

        # calculate the optimal step size for this iteration of the search
        # up to 0.5*alpha_min (multiplying by 0.5 to reduce the chance of hitting
        # the feasible direction boundary)
        alpha_star = step_length(0.5*alpha_min, w, b, dw, db, x_train, y_train, C)
        if alpha_star <= 0:
            logging.warning('NON-POSITIVE STEP LENGTH: {}'.format(alpha_star))
            break
        
        # calculate the objective at the previous step
        obj_old = calc_objective(w, b, x_train, y_train, C)

        # update w, b with the gradient and step size
        w = w + alpha_star*dw
        b = b + alpha_star*db

        # calculate the objective after taking the step
        obj_new = calc_objective(w, b, x_train, y_train, C)

        # log the objective every 50 iterations
        if i % 50 == 0:
            logging.info('ITERATION {}, OBJECTIVE {}'.format(i, obj_new))
        
        # check for convergence
        if abs(obj_new - obj_old) < TOLERANCE:
            logging.info('CONVERGED AFTER {} ITERATIONS'.format(i))
            break

    # check if we exited the loop after the maximum number of iterations
    if i == MAX_ITER-1:
        logging.warning('WARNING: MAXIMUM NUMBER OF ITERATIONS REACHED')


    logging.info('FINAL OBJECTIVE: {}'.format(calc_objective(w, b, x_train, y_train, C)))
    
    return (w, b)

def test_precision(w, b, x_test, y_test):
    """Measure the precision and number of labeled examples above the decision
    boundary given a parameter set (w, b) and test examples & labels.

    Parameters
    ----------
    w : ndarray
        Array of feature weights from the model
    b : float
        Model bias term
    x_test : ndarray
        Matrix of the test examples
    y_test : ndarray
        Array of labels for the test examples in [-1, 1]

    Return value
    ------------
    len(labeled) : int
        Number of labeled examples above the decision boundary
    prec : float
        Precision among the predicted positives
    """

    # indices of predicted positives (L)
    L, E, R = LER(w, b, x_test)

    # replace -1's with 0's in y_test to make precision calcuation easier
    y_test_zero = y_test.copy()
    y_test_zero[y_test_zero == -1] = 0

    # predicted labels
    pred = np.zeros_like(y_test_zero)
    pred[L] = 1

    # not all examples above the threshold might be labeled...
    df = pd.DataFrame({'pred': pred, 'obs': y_test_zero})
    labeled = df.loc[(df['pred']==1) & (df['obs'].notnull())]   # keep labeled examples (non-null obs) above threshold (pred==1)
    prec = 1.0*labeled['obs'].sum()/len(labeled)

    # return the number of labeled examples in top k and precision
    return (len(labeled), prec)


def prep_data(train_matrix_file, test_matrix_file, scale_data, data_type):
    """Read in and scale data for the TTK model. Data should be either in CSV or HDF5
    format and is expected to contain a header and labels in a column named `outcome`.
    Entity IDs and dates in `entity_id` and `as_of_date` columns.

    Parameters
    ----------
    train_matrix_file : string
        Path to the file with the training data in either CSV or HDF5 format
        should contain a header and labels in an `outcome` column
    test_matrix_file : string
        Path to the file with the test data in either CSV or HDF5 format
        should contain a header and labels in an `outcome` column
    scale_data : string
        Method for scaling the data: 'z' will scale to mean 0, stdev 1; 
        'minmax' will scale to range [0,1]; 'none' will skip scaling
    data_type : string
        Specify the type of data to read in: 'csv', 'hdf5', or 'infer'
        ('infer' will determine the data type based on the extension)

    Return value
    ------------
    x_train : ndarray
        Matrix of the training examples
    y_train : ndarray
        Array of labels for the training examples in [-1, 1]
    x_test : ndarray
        Matrix of the test examples
    y_test : ndarray
        Array of labels for the test examples in [-1, 1]
    """

    # if data_type is 'infer', determine the data type from file extension
    if data_type == 'infer':
        if train_matrix_file[-3:].lower() == 'csv':
            data_type = 'csv'
        if train_matrix_file[-2:].lower() == 'h5':
            data_type = 'hdf5'

    # read in the data using the appropriate pandas method
    if data_type == 'csv':
        train_mat = pd.read_csv(train_matrix_file)
        test_mat = pd.read_csv(test_matrix_file)
    elif data_type == 'hdf5':
        train_mat = pd.read_hdf(train_matrix_file)
        test_mat = pd.read_hdf(test_matrix_file)
    else:
        raise ValueError('Could not read input data type {}'.format(data_type))


    logging.debug('Train Columns Head: {}'.format(train_mat.columns.values[:10]))
    logging.debug('Test Columns Head: {}'.format(test_mat.columns.values[:10]))

    logging.debug('Train Columns Tail: {}'.format(train_mat.columns.values[-10:]))
    logging.debug('Test Columns Tail: {}'.format(test_mat.columns.values[-10:]))

    # check that the train and test matrices have the same columns
    if list(train_mat.columns) != list(test_mat.columns):
        raise ValueError('Train and Test Matrix Column Mismatch')

    # ensure the outcome variable is mapped to -1, 1 for SVM
    train_mat['outcome'] = train_mat['outcome'].map({1: 1, 0: -1, -1: -1})
    test_mat['outcome'] = test_mat['outcome'].map({1: 1, 0: -1, -1: -1})

    assert(list(train_mat.columns.values) == list(test_mat.columns.values))

    # drop label, ID, and as of date columns from the matrices
    dropcols = ['outcome']
    for col in ['entity_id', 'officer_id', 'as_of_date']:
        if col in train_mat.columns:
            dropcols.append(col)

    # not sure why some columns are coming in as objects which are oddly getting
    # coerced to decimal.Decimal rather than floats when we create x_train and x_test
    # but let's force these to floats to be sure.
    for col in train_mat.drop(dropcols, axis=1).columns:
        if (train_mat.dtypes[col] == 'object') or (test_mat.dtypes[col] == 'object'):
            train_mat[col] = train_mat[col].astype(np.float64)
            test_mat[col] = test_mat[col].astype(np.float64)

    # split out the train & test features and labels
    y_train = train_mat['outcome'].values
    x_train = train_mat.drop(dropcols, axis=1).values

    y_test = test_mat['outcome'].values
    x_test = test_mat.drop(dropcols, axis=1).values

    logging.debug('x_train.shape: {}'.format(x_train.shape))
    logging.debug('x_test.shape: {}'.format(x_test.shape))

    # apply the desired scaling to the features
    if scale_data != 'none':
        if scale_data == 'z':
            logging.debug('Scaling Data with standard scaler...')
            scaler = StandardScaler()
        elif scale_data == 'minmax':
            logging.debug('Scaling Data with min-max scaler...')
            scaler = MinMaxScaler()
        else:
            raise ValueError('Invalid scale parameter {}'.format(scale_data))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    return (x_train, y_train, x_test, y_test)


def main():

    args = parser.parse_args()

    if args.k is None and args.kfrac is None:
        raise ValueError('Must specify either k or kfrac')

    if args.scaledata not in ['none', 'z', 'minmax']:
        raise ValueError('scaledata must be one of: none, z, minmax')

    if args.datatype not in ['infer', 'csv', 'hdf5']:
        raise ValueError('datatype must be one of: infer, csv, hdf5')

    # configure logging
    log_filename = 'logs/ttk_{}_C{}_k{}_Scale-{}'.format(
        str(datetime.datetime.now()).replace(' ', '_').replace(':', ''),
        args.C, args.k if args.k else args.kfrac, args.scaledata
    )
    if args.verbose:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(process)d %(levelname)s: %(message)s', 
        level=logging_level,
#        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
        filename=log_filename
    )

    # read in and scale the data and determine the value of k to use
    x_train, y_train, x_test, y_test = prep_data(args.trainmat, args.testmat, args.scaledata, args.datatype)
    k = args.k if args.k else int(round(args.kfrac * x_test.shape[0]))

    logging.info('Running TTK with C={}, k={}, scale_data={}, and MAX_ITER={}'.format(args.C, k, args.scaledata, args.maxiter))
    logging.info('Using train matrix {}'.format(args.trainmat))
    logging.info('Using test matrix {}'.format(args.testmat))

    # run the TTK model
    w, b = run_ttk(x_train, y_train, x_test, k, C=args.C, MAX_ITER=args.maxiter)

    # Dump the resulting w, b, and input parameters to a pickle file
    pickle_file = 'pickles/ttk_{}_C{}_k{}_Scale-{}'.format(
        str(datetime.datetime.now()).replace(' ', '_').replace(':', ''),
        args.C, args.k if args.k else args.kfrac, args.scaledata
    )
    pf = open(pickle_file, 'wb')
    pickle.dump((w, b, k, args.C, args.maxiter), pf)
    pf.flush()
    pf.close()

    # check the precision for the resulting model
    num_labeled, precision = test_precision(w, b, x_test, y_test)

    logging.info('Precision at top {}: {} with {} labeled above threshold'.format(k, precision, num_labeled))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the transductive top k model from Liu, et al. (2015): arXiv:1510.05976')
    parser.add_argument("-a", "--trainmat", help="Training Matrix File Location",
                        action="store", required=True)
    parser.add_argument("-b", "--testmat", help="Test Matrix File Location",
                        action="store", required=True)
    parser.add_argument("-k", "--k", help="Top k to optimized to (as integer)",
                        action="store", type=int)
    parser.add_argument("-f", "--kfrac", help="Top k to optimized to (as fraction)",
                        action="store", type=float)
    parser.add_argument("-C", "--C", help="Slack Penalty Parameter",
                        action="store", default=1.0, type=float)
    parser.add_argument("-i", "--maxiter", help="max number of iterations to run",
                        action="store", default=5000, type=int)
    parser.add_argument("-s", "--scaledata", help="Scale data: 'z' will scale to mean 0, stdev 1; 'minmax' will scale to range [0,1]; default no scaling",
                        action="store", default='none', type=str.lower)
    parser.add_argument("-t", "--datatype", help="data type: 'csv', 'hdf5', or 'infer' (default)",
                        action="store", default='infer', type=str.lower)
    parser.add_argument("-v", "--verbose", help="Enable debug logging",
                        action="store_true")

    main()




