import matplotlib.pyplot as plt
import numpy as np

class NeuralNet():

    def __init__(self, input_size, hidden_size, output_size, std=5):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def process(self,X,Y,gradients=True):
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

        dataloss = -np.log(probs[range(N),Y])
        dataloss = np.sum(dataloss)/N
        print dataloss

        if not gradients:
            return dataloss

        delta3 = probs
        delta3[range(N),Y.reshape(1,N)] -= 1

        db2 = np.sum(delta3,axis=0)
        dW2 = np.dot(a2.T,delta3)

        delta2 = np.multiply(np.dot(delta3,W2.T) , (z2>0))
        dW1 = np.dot(X.T,delta2)
        db1 = np.sum(delta2,axis=0)

        grads = {}
        grads['dW1'] = dW1
        grads['dW2'] = dW2
        grads['db1'] = db1
        grads['db2'] = db2

        print "probs: "+str(probs.shape)
        print "X : "+str(X.shape)

        print "W1 : "+str(W1.shape)
        print "b1 : "+str(b1.shape)
        print "z2 : "+str(z2.shape)

        print "W2 : "+str(W2.shape)
        print "b2 : "+str(b2.shape)
        print "z3 : "+str(z3.shape)
        print
        print "dW2 : "+str(dW2.shape)
        print "delta2 : "+str(delta2.shape)
        print "dW1 : "+str(dW1.shape)
        print "db1 : "+str(db1.shape)
        print "db2 : "+str(db2.shape)

        return dataloss, grads
        # dW2 = np.sum(a2,axis=0)

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

        dataloss = -np.log(probs[range(N),Y])
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
        # loss1 = self.process (X,Y,gradients = False)
        # loss2 = self.process (X+h,Y,gradients = False)

np.random.seed(0)
NN = NeuralNet(2,3,3)
X = np.random.randn(100,2)
Y = np.random.randint(3,size=(100,1))
loss,g = NN.process(X,Y)
print g
grads = NN.gradient_check(X,Y)
print "dW1: "
print grads[0]
print "db1: "
print grads[1]
print "dW2: "
print grads[2]
print "db2: "
print grads[3]

print"W1:"
print g['dW1']-grads[0]
print"b2:"
print g['db2']-grads[3]
