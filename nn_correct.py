import numpy as np
import matplotlib.pyplot as plt


class NeuralNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.
    In other words, the network has the following architecture:
    input - fully connected layer - ReLU - fully connected layer - softmax
    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:
        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)
        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.
        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. This parameter is optional; if it
        is not passed then we only return scores, and if it is passed then we
        instead return the loss and gradients.
        - reg: Regularization strength.
        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].
        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
        samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None

        #############################################################################
        y1 = X.dot(W1) + b1 #(N,H) + (H)
        h1 = np.maximum(0, y1)
        y2 = h1.dot(W2) + b2
        scores = y2

        if y is None:
            return scores

            # Compute the loss
        loss = None

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

        # select all corresponding log likelihood score for each class
        # N being the number of sample and take the mean of it
        corect_logprobs = -np.log(probs[range(N), y])
        loss = np.sum(corect_logprobs) / N #np.mean()
        reg_loss = 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss += reg_loss

        grads = {}

        dy2 = probs
        dy2[range(N),y] -= 1

        # don't forget the constant 1/N (derivate of constant is constant...)
        dy2 /= N

        # y2 = h1.dot(W2) + b2
        dW2 = h1.T.dot(dy2)

        # chaine rule
        dh1 = dy2.dot(W2.T)

        # gradient of Relu
        dy1 = dh1 * (y1 >= 0)

        # y1 = X.dot(W1) + b1
        dW1 = X.T.dot(dy1)

        db1 = np.sum(dy1, axis=0)
        db2 = np.sum(dy2, axis=0)

        # Regularization
        dW1 += reg * W1
        dW2 += reg * W2


        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['b1'] = db1
        grads['b2'] = db2

        return loss, grads


    def train(self, X, y,
            learning_rate=1, learning_rate_decay=0.95,
            reg=1e-5, num_iters=2000,
            batch_size=200, verbose=True):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in xrange(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

          # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
            # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                train_acc_history.append(train_acc)
                loss_history.append(loss)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
        }

    def predict(self, X):
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        z1 = X.dot(self.params['W1']) + self.params['b1']
        a1 = np.maximum(0, z1) # pass through ReLU activation function
        scores = a1.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred

np.random.seed(0)

np.random.seed(0)
NN = NeuralNet(2,100,3)

N = 300
K = 3
D = 2
X = np.zeros((N*K,D))
Y = np.zeros(N*K, dtype='uint8')
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  Y[ix] = j

# Y=Y.reshape(N*K,1)




print (NN.predict(X))
print (Y)

history = NN.train(X,Y)

plt.plot(history['train_acc_history'])
# plt.show()
# for g in grads:
    # print g, grads[g]
    # print
