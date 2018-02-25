import numpy as np


def linear_forward(X, W, b):
    """
    Computes the forward pass for a linear (fully-connected) layer.

    The input X has shape (N, d_1, ..., d_K) and contains N samples with each
    example X[i] has shape (d_1, ..., d_K). Each input is reshaped to a
    vector of dimension D = d_1 * ... * d_K and then transformed to an output
    vector of dimension M.

    Args:
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: A numpy array of shape (D, M) of weights, with D = d_1 * ... * d_K
    - b: A numpy array of shape (M, ) of biases

    Returns:
    - out: linear transformation to the incoming data
    """
    out = None
    """
    TODO: Implement the linear forward pass. Store your result in `out`.
    Tip: Think how X needs to be reshaped.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################

    X = np.reshape(X,(-1,W.shape[0]))

    out = np.add(np.dot(X,W), b)
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out

def linear_backward(dout, X, W, b):
    """
    Computes the backward pass for a linear (fully-connected) layer.

    Args:
    - dout: Upstream derivative, of shape (N, M)
    - X: A numpy array of shape (N, d_1, ..., d_K) incoming data
    - W: A numpy array of shape (D, M) of weights, with D= d_1 * ... * d_K
    - b: A numpy array of shape (M, ) of biases

    Returns (as tuple):
    - dX: A numpy array of shape (N, d_1, ..., d_K), gradient with respect to x
    - dW: A numpy array of shape (D, M), gradient with respect to W
    - db: A nump array of shape (M,), gradient with respect to b
    """
    dX, dW, db = None, None, None
    """
    TODO: Implement the linear backward pass. Store your results of the
    gradients in `dX`, `dW`, `db`.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
#    print("linear backwards incoming: ")
#    print ("X shape: ",X.shape)
#    print("W shape: ",W.shape)
#    print("dout shape: ",dout.shape)
    
    dX = dout.dot(W.T).reshape(X.shape)
#    print(dX)
    dW = X.reshape(X.shape[0], -1).T.dot(dout)
#    print(dW)
    db = np.sum(dout, axis=0)
#     print(db)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

    return dX, dW, db

def relu_forward(X):
    """
    Computes the forward pass for rectified linear unit (ReLU) layer.
    Args:
    - X: Input, an numpy array of any shape
    Returns:
    - out: An numpy array, same shape as X
    """
    out = None
    """
    TODO: Implement the ReLU forward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
#    print("relu called: input ",X)
    out = X.copy()  # Must use copy in numpy to avoid pass by reference.
    out[out < 0] = 0
    out = np.maximum(0, X)
#    print("relu called: output ",X)
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out

def relu_backward(dout, X):
    """
    Computes the backward pass for rectified linear unit (ReLU) layer.
    Args:
    - dout: Upstream derivative, an numpy array of any shape
    - X: Input, an numpy array with the same shape as dout
    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the ReLU backward pass. Store your result in out.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    dX = (X > 0) * dout

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX

def dropout_forward(X, p=0.5, train=True, seed=42):
    """
    Compute f
    Args:
    - X: Input data, a numpy array of any shape.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns (as a tuple):
    - out: Output of dropout applied to X, same shape as X.
    - mask: In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    # print("entered dropout forward")
    out = None
    mask = None
    if seed:
        np.random.seed(seed)
    """
    TODO: Implement the inverted dropout forward pass. Make sure to consider
    both train and test case. Pay attention scaling the activation function.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    #    we only use the dropout mask if we are in training. Otherwise, mask = None.
    #    We add the scaling factor(1/p) as part of the inverted dropout.
    if train==True:
        # print("entered true")
        q=1-p
        mask = np.random.binomial(1,p=q,size=X.shape) * (1/(q))
        out = X * mask
    else:
        out = X

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return out, mask

def dropout_backward(dout, mask, p=0.5, train=True):
    """
    Compute the backward pass for dropout
    Args:
    - dout: Upstream derivative, a numpy array of any shape
    - mask: In training mode, mask is the dropout mask that was used to
      multiply the input; in test mode, mask is None.
    - p: Dropout parameter. We drop each neuron output with probability p.
    Default is p=0.5.
    - train: Mode of dropout. If train is True, then perform dropout;
      Otherwise train is False (= test mode). Default is train=True.

    Returns:
    - dX: A numpy array, derivative with respect to X
    """
    dX = None
    """
    TODO: Implement the inverted backward pass for dropout. Make sure to
    consider both train and test case.
    """
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    if train == True:
        dX = dout * mask
    else:
        dX = dout

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return dX
