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
