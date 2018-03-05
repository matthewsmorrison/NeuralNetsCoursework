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
    normalisation!
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    
    N = logits.shape[0]

    # Calculate softmax
    stable_logits = logits - np.max(logits)
    softmax = (np.exp(stable_logits)) / np.sum(np.exp(stable_logits),axis=1)[:,None]
    
    # Now calculate the loss

    loss = -np.log(softmax[np.arange(len(softmax)), y]).sum()
    loss /= N

    # Now calculate the gradient
    dlogits = np.copy(softmax)
    dlogits[np.arange(N),y] -=1
    dlogits /= N

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return loss, dlogits
