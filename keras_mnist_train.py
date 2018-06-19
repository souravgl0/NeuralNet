import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from sklearn.preprocessing import OneHotEncoder
import keras.backend as K

reg = 0

model = Sequential()
model.add(Dense(units=800,
                activation='relu',
                input_shape=(784,),
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=None),
                kernel_regularizer=regularizers.l2(reg)))
# model.add(Dense(units=300,
                # activation='relu',
                # kernel_regularizer=regularizers.l2(reg)))
model.add(Dense(10,
                activation='softmax',
                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1e-4, seed=None),
                kernel_regularizer=regularizers.l2(reg)))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# print model.get_weights()

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
    train_data[train_data>50]=255
    train_data[train_data<=50]=0
    test_data[test_data>100]=255
    test_data[test_data<100]=0
    train_data /=255
    test_data /=255
    return train_data,test_data

X_train,Y_train,X_val,Y_val = getdata()
# X_train,X_val = standardise(X_train,X_val)
# print X_train[0]
# plt.imshow(X_train[0].reshape(28,28))
# plt.show()

onehotencoder = OneHotEncoder(sparse=False)
Y_train = onehotencoder.fit_transform(Y_train)
Y_val = onehotencoder.fit_transform(Y_val)

history = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=10,verbose=1)
score = model.evaluate(X_train,Y_train)
print "Train:",score[1]
score = model.evaluate(X_val,Y_val)
print "Validation:",score[1]

# model.save("MNIST_model_new.keras")

# with open('model_history','w') as f:
    # pickle.dump(history.history,f)

plt.subplot(2,1,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','validation'])

plt.subplot(2,1,2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(['train','validation'])

# plt.show()
