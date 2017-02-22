import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
    
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_encoded = make_OneHot(y)
  scores = X.dot(W)
  prob = softmax(scores)

  loss = (-1 / float(X.shape[0])) * np.sum(y_encoded * np.log(prob)) + (reg/2)*np.sum(W*W)
  # scores = W.dot(X) # [K, N]
  # # Shift scores so that the highest value is 0
  # scores -= np.max(scores)
  # scores_exp = np.exp(scores)
  # correct_scores_exp = scores_exp[y, xrange(num_train)] # [N, ]
  # scores_exp_sum = np.sum(scores_exp, axis=0) # [N, ]
  # loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
  # loss /= num_train
  # loss += 0.5 * reg * np.sum(W * W)
  dW = (-1 / float(X.shape[0])) * np.dot(X.T,(y_encoded - prob)) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_encoded = make_OneHot(y)
  scores = X.dot(W)
  prob = softmax(scores)
  loss = (-1 / float(X.shape[0])) * np.sum(y_encoded * np.log(prob)) + (reg/2)*np.sum(W*W)
  dW = (-1 / float(X.shape[0])) * np.dot(X.T,(y_encoded - prob)) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm
def make_OneHot(y):
    C=np.max(y)
    rtn = np.zeros([y.shape[0],C+1])
    rtn[np.arange(y.shape[0]),y]=1
    return rtn
