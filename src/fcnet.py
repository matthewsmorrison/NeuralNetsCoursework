import numpy as np

from src.classifiers import softmax
from src.layers import (linear_forward, linear_backward, relu_forward,
                        relu_backward, dropout_forward, dropout_backward)



def random_init(n_in, n_out, weight_scale=5e-2, dtype=np.float32):
    """
    Weights should be initialized from a normal distribution with standard
    deviation equal to weight_scale and biases should be initialized to zero.

    Args:
    - n_in: The number of input nodes into each output.
    - n_out: The number of output nodes for each input.
    """
    W = None
    b = None
    ###########################################################################
    #                           BEGIN OF YOUR CODE                            #
    ###########################################################################
    W = np.random.normal(scale = weight_scale,size=(n_in,n_out))
    b = np.zeros(n_out,dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return W, b



class FullyConnectedNet(object):
    """
    Implements a fully-connected neural networks with arbitrary size of
    hidden layers. For a network with N hidden layers, the architecture would
    be as follows:
    [linear - relu - (dropout)] x (N - 1) - linear - softmax
    The learnable params are stored in the self.params dictionary and are
    learned with the Solver.
    """
    def __init__(self, hidden_dims, input_dim=32*32*3, num_classes=10,
                 dropout=0, reg=0.0, weight_scale=1e-2, dtype=np.float32,
                 seed=None):
        """
        Initialise the fully-connected neural networks.
        Args:
        - hidden_dims: A list of the size of each hidden layer
        - input_dim: A list giving the size of the input
        - num_classes: Number of classes to classify.
        - dropout: A scalar between 0. and 1. determining the dropout factor.
        If dropout = 0., then dropout is not applied.
        - reg: Regularisation factor.

        """
        self.dtype = dtype
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.use_dropout = True if dropout > 0.0 else False
        if seed:
            np.random.seed(seed)
        self.params = dict()
        """
        TODO: Initialise the weights and bias for all layers and store all in
        self.params. Store the weights and bias of the first layer in keys
        W1 and b1, the weights and bias of the second layer in W2 and b2, etc.
        Weights and bias are to be initialised according to the Xavier
        initialisation (see manual).
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        w = "W"
        b = "b"
#        for input to first hidden layer
        W,B = random_init(input_dim, hidden_dims[0],weight_scale,dtype)
        self.params[w+"1"] = W
        self.params[b+"1"] = B
#       for between the hidden layers
        for i in range(1,self.num_layers):

            if i == self.num_layers-1:
            #        for last hidden layer to output layer
                W,B = random_init(hidden_dims[i-1], num_classes,weight_scale,dtype)
            else:
                W,B = random_init(hidden_dims[i-1], hidden_dims[i],weight_scale,dtype)
            self.params[w+str(i+1)] = W
            self.params[b+str(i+1)] = B



#        for i in self.params:
#            print(i, self.params[i])

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # When using dropout we need to pass a dropout_param dictionary to
        # each dropout layer so that the layer knows the dropout probability
        # and the mode (train / test). You can pass the same dropout_param to
        # each dropout layer.
        self.dropout_params = dict()
        if self.use_dropout:
            self.dropout_params = {"Train": True, "p": dropout}
            if seed is not None:
                self.dropout_params["seed"] = seed
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        Args:
        - X: Input data, numpy array of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        Returns:
        If y is None, then run a test-time forward pass of the model and
        return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward pass
        and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
        parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        X = X.astype(self.dtype)
        linear_cache = dict()
        relu_cache = dict()
        dropout_cache = dict()
        """
        TODO: Implement the forward pass for the fully-connected neural
        network, compute the scores and store them in the scores variable.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################

        # Training time

        # let's train the forward pass according to [linear - relu - (dropout)] x (N - 1) - linear - softmax
        curr_act = X
        activations = [X]
        mask_cache = dict()

        epsilon = 0.00000000001

        #for the hidden layers except the last one
        for i in range(self.num_layers-1):
            linear_cache["L"+str(i+1)] = curr_act
            curr_act = linear_forward(curr_act, W=self.params["W"+str(i+1)],b=self.params["b"+str(i+1)])
            print(i, curr_act.shape)
            relu_cache["R"+str(i+1)] = curr_act
            curr_act = relu_forward(curr_act)
            print(curr_act.shape)
#            print("curr_act: ",curr_act)
            if self.use_dropout:
                dropout_cache["D"+str(i+1)] = curr_act
                curr_act, mask = dropout_forward(curr_act,p=self.dropout_params["p"],train=self.dropout_params["Train"],seed=self.dropout_params["seed"])
                mask_cache["R"+str(i+1)] = mask
            activations.append([curr_act])


        # Then for the final hidden layer, which feeds the output classes
        linear_cache["L"+str(self.num_layers)] = curr_act
        curr_act = linear_forward(curr_act, W=self.params["W"+str(self.num_layers)],b=self.params["b"+str(self.num_layers)])
        print(curr_act.shape)
        #print(curr_act)
        #if self.use_dropout:
        #    dropout_cache["D"+str(self.num_layers)] = curr_act
        #    curr_act, mask = dropout_forward(curr_act,p=self.dropout_params["p"],train=self.dropout_params["Train"],seed=self.dropout_params["seed"])
        #    mask_cache["R"+str(self.num_layers)] = mask
        scores = curr_act+epsilon
        #print(curr_act)
#        print("activations: ",activations)
#        print("curr act size: ",curr_act.shape)
#        print("curr act: ",curr_act)

#        print("grads size: ",grads.shape)
#        print("grads: ",grads)
#        stable_logits = curr_act - np.max(curr_act)
#        scores = (np.exp(curr_act)) / np.sum(np.exp(curr_act),axis=1)[:,None]
#        print("curr act after softmax: ",scores)
#        print("loss: ",loss)

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        loss, grads = 0, dict()

        """
        TODO: Implement the backward pass for the fully-connected net. Store
        the loss in the loss variable and all gradients in the grads
        dictionary. Compute the loss with softmax. grads[k] has the gradients
        for self.params[k]. Add L2 regularisation to the loss function.
        NOTE: To ensure that your implementation matches ours and you pass the
        automated tests, make sure that your L2 regularization includes a
        factor of 0.5 to simplify the expression for the gradient.
        """
        #######################################################################
        #                           BEGIN OF YOUR CODE                        #
        #######################################################################
        loss,deltas = softmax(curr_act,y)

        for i in range(self.num_layers):
#            add in the L2 regularisation terms for each of the layers
            loss += 0.5 * self.reg * np.sum(self.params["W"+str(i+1)]) * np.sum(self.params["W"+str(i+1)])

        #backpropagate through the last layer
        s = str(self.num_layers)
        #if self.use_dropout:
        #        deltas = dropout_backward(deltas,p=self.dropout_params["p"],train=self.dropout_params["Train"], mask = mask_cache["R"+s])
        dX,dW,dB = linear_backward(dout=deltas,X=linear_cache["L"+s],W=self.params["W"+s],b=self.params["b"+s])
        grads["W" + s] = dW + self.reg * self.params["W"+s]
        grads["b" + s] = dB

#        now backpropagate through all the remaining layers
        for i in range(self.num_layers-1,0,-1):
            s = str(i)
#            first through the dropout backwards
            if self.use_dropout:
               dX  = dropout_backward(dX,p=self.dropout_params["p"],train=self.dropout_params["Train"], mask = mask_cache["R"+s])
#        then through the relu backwards
            dX = relu_backward(dX,relu_cache["R"+s])
#       then through the linear transform backwards
            dX,dW,dB = linear_backward(dout=dX,X=linear_cache["L"+s],W=self.params["W"+s],b=self.params["b"+s])
            grads["W" + s] = dW + self.reg * self.params["W"+s]
            grads["b" + s] = dB

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
        return loss, grads


#model = FullyConnectedNet(hidden_dims = [5,3,2],input_dim=15)
# N, D, H1, H2, C = 2, 15, 2, 3, 10
# X = np.random.randn(N, D)
# y = np.random.randint(C, size=(N,))
#
# model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C, dtype=np.float64)
# loss, grads = model.loss(X, y)
# print("loss: ",loss)
# print("grads: ",grads)
