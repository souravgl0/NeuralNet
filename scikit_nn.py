import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

np.random.seed(0)


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

reg = 1e-2
hdn = 10
bsize = 200
learning_rate = 0.01
it =    2000
NN = MLPClassifier(solver='sgd', alpha=reg, hidden_layer_sizes = (hdn),
                   batch_size=bsize, random_state =0, learning_rate_init = learning_rate,
                   max_iter=it, verbose=True)

NN.fit(X,Y)

print (NN.predict(X)==Y.reshape(1,N*K)).mean()
# print (NN.predict(X)==Y).mean()

plt.plot(NN.loss_curve_)
plt.show()

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
# Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], NN.params['W1']) + NN.params['b1']), NN.params['W2']) + NN.params['b2']
# Z = np.argmax(Z, axis=1)
Z = NN.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
