import matplotlib.pyplot as plt
import numpy as np

class NeuralNet():

    def __init__(self, input_size, hidden_size, output_size, std=1):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def process(self,X,Y,reg=0.0,gradients=True):
        """
        X : (N,D) N = Number of Samples, D = input_size
        Y : (N,1) N = Number of Samples

        if Y is None, return probs after forward pass
        else
        Return loss, gradients if gradients = True
        Return loss otherwise
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        N = X.shape[0]
        D = X.shape[1]

        z2 = np.dot(X,W1)
        z2 = z2 + b1
        #ReLu
        a2 = np.maximum(z2,0)
        z3 = np.dot(a2,W2)
        z3 = z3 + b2
        # SOFTMAX

        probs = np.exp(z3)
        expsum = np.sum(probs,axis=1).reshape(N,1)
        probs = probs/expsum

        if Y is None:
            return probs

        dataloss = -np.log(probs[range(N),Y.reshape(1,N)])
        dataloss = np.sum(dataloss)/N

        regloss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
        loss = dataloss + regloss
        # print dataloss

        if not gradients:
            return loss

        delta3 = probs
        delta3[range(N),Y.reshape(1,N)] -= 1
        delta3 /= N

        db2 = np.sum(delta3,axis=0)
        dW2 = np.dot(a2.T,delta3)

        delta2 = np.multiply(np.dot(delta3,W2.T) , (z2>0))
        dW1 = np.dot(X.T,delta2)
        db1 = np.sum(delta2,axis=0)

        grads = {}
        grads['dW1'] = dW1 + reg * W1
        grads['dW2'] = dW2 + reg * W2
        grads['db1'] = db1
        grads['db2'] = db2

        return loss, grads

    def train(self, X, Y, reg=1e-5, num_iters=2000,
              batch_size=200, learning_rate=0.8,
              learning_decay=1):

        N = X.shape[0]
        iterations_per_epoch = max(N / batch_size, 1)

        loss_history = []
        train_acc_history = []

        for it in xrange(num_iters):
            inds = np.random.choice(N,batch_size)
            X_batch = X[inds,:]
            Y_batch = Y[inds,:]

            loss,grads = self.process(X_batch,Y_batch,reg=reg)

            self.params['W1'] -= learning_rate * grads['dW1']
            self.params['W2'] -= learning_rate * grads['dW2']
            self.params['b1'] -= learning_rate * grads['db1']
            self.params['b2'] -= learning_rate * grads['db2']

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == Y_batch).mean()
                loss = self.process(X,Y,gradients=False)
                loss_history.append(loss)
                train_acc_history.append(train_acc)
            if it % 100 == 0:
                learning_rate *= learning_decay
                print "Iteration: %d/%d loss: %f , Train Accuracy: %f" %(it, num_iters, loss, train_acc)
                print learning_rate
                # if len(loss_history) > 1 and abs(loss_history[-2] - loss) < 1e-8:
                    # print loss_history[-2]
                    # print loss
                    # break
                #calculate Accuracy Training,Validation

        return loss_history, train_acc_history

    def predict(self, X):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        N = X.shape[0]

        z2 = np.dot(X,W1)
        z2 = z2 + b1
        #ReLu
        a2 = np.maximum(z2,0)

        z3 = np.dot(a2,W2)
        z3 = z3 + b2
        # SOFTMAX

        probs = np.exp(z3)
        expsum = np.sum(probs,axis=1).reshape(N,1)
        probs = probs/expsum

        return np.argmax(probs,axis=1).reshape(N,1)


    def f(self,X,Y,params):
        W1,b1,W2,b2 = params

        N = X.shape[0]
        D = X.shape[1]

        z2 = np.dot(X,W1)
        z2 = z2 + b1
        #ReLu
        a2 = np.maximum(z2,0)

        z3 = np.dot(a2,W2)
        z3 = z3 + b2
        # SOFTMAX
        probs = np.exp(z3)
        expsum = np.sum(probs,axis=1).reshape(N,1)
        probs = probs/expsum

        dataloss = -np.log(probs[range(N),Y.reshape(1,N)])
        dataloss = np.sum(dataloss)/N

        return dataloss

    def gradient_check(self,X,Y):
        h=1e-5
        params = [self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']]

        grads = []
        for i,param in enumerate(params):
            shape = param.shape
            grad = np.zeros(shape)

            it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                orig = param[it.multi_index]
                param[it.multi_index] = orig + h
                params[i] = param
                pval = self.f(X,Y,params)

                param[it.multi_index] = orig - h
                params[i] = param
                nval = self.f(X,Y,params)

                param[it.multi_index] = orig
                params[i]=param

                grad[it.multi_index] = (pval-nval)/(2.0*h)
                it.iternext()

            grads.append(grad)

        return grads


def plot(X,Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

np.random.seed(0)
NN = NeuralNet(2,10,3)

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

Y=Y.reshape(N*K,1)


loss_history,train_acc_history = NN.train(X,Y)

# plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.plot(loss_history)
# plt.show()
print (NN.predict(X)==Y).mean()



h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], NN.params['W1']) + NN.params['b1']), NN.params['W2']) + NN.params['b2']
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
plt.draw()
