import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from neural_net import NeuralNet

def getdata(size=1):
    train_df = pd.read_csv("MNIST_data/train_data.csv")
    cv_df = pd.read_csv("MNIST_data/cv_data.csv")

    dsize = train_df.shape[0]
    size = size * dsize
    X_train = train_df.iloc[:size,1:].astype('float32')
    Y_train = train_df.iloc[:size,0:1]
    X_val = cv_df.iloc[:,1:].astype('float32')
    Y_val = cv_df.iloc[:,0:1]

    return X_train.as_matrix(),Y_train.as_matrix(),X_val.as_matrix(),Y_val.as_matrix()

def standardise(train_data, test_data):
    train_data /= 255
    test_data /= 255
    return train_data,test_data

X_train,Y_train,X_val,Y_val = getdata()
X_train,X_val = standardise(X_train,X_val)
# print X_train[1].reshape(28,28)
# plt.imshow(X_train[1].reshape(28,28))
# plt.show()
# exit()
model = NeuralNet(784,1000,10)

loss_history,train_acc_history = model.train(X_train,Y_train,
                                             reg=0,num_iters=2000,
                                             batch_size=32,learning_rate=0.1)

plt.subplot(2,1,1)
plt.plot(loss_history)
plt.title('Loss History')
plt.ylabel('Loss')

plt.subplot(2,1,2)
plt.plot(train_acc_history)
plt.title('Train Accuracy History')
plt.ylabel('Train Accuracy')
plt.show()

print "Train Accuracy:",(model.predict(X_train)==Y_train).mean()
print "Validation Accuracy:",(model.predict(X_val)==Y_val).mean()
