import numpy as np


def softmax(logits, y):
    """
    Computes the loss and gradient for softmax classification.

    Args:
    - logits: A numpy array of shape (N, C)
    - y: A numpy array of shape (N,). y represents the labels corresponding to
    logits, where y[i] is the label of logits[i], and the value of y have a
    range of 0 <= y[i] < C

    Returns (as a tuple):
    - loss: Loss scalar
    - dlogits: Loss gradient with respect to logits
    """
    loss, dlogits = None, None
    """
    TODO: Compute the softmax loss and its gradient using no explicit loops
    Store the loss in loss and the gradient in dW. If you are not careful
    here, it is easy to run into numeric instability. Don't forget the
    regularization!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    reg = 1

    number_train = logits.shape[0]
    shift_logits = logits - np.max(logits, axis=1).reshape(-1,1)
    softmax_output = np.exp(shift_logits) / np.sum(np.exp(shift_logits),axis=1).reshape(-1,1)
    loss = -np.sum(np.log(softmax_output[range(number_train),list(y)]))
    loss /= number_train
    loss += 0.5 * reg * np.sum(logits * logits)

    dS = softmax_output.copy()
    dS[range(number_train),list(y)] += -1
    dlogits = (logits.T).dot(dS)
    dlogits = dlogits/number_train + reg * logits

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
