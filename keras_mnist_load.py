import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.preprocessing import OneHotEncoder

model = keras.models.load_model("MNIST_model_reg1e-4.keras")

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
    train_data /=255
    test_data /=255
    return train_data,test_data

X_train,Y_train,X_val,Y_val = getdata()
X_train,X_val = standardise(X_train,X_val)

onehotencoder = OneHotEncoder(sparse=False)
Y_train = onehotencoder.fit_transform(Y_train)
Y_val = onehotencoder.fit_transform(Y_val)

print model.metrics_names
score = model.evaluate(X_val,Y_val)
print "Validation : ",score
score = model.evaluate(X_train,Y_train)
print "Training : ",score

test_data = pd.read_csv("MNIST_data/test.csv")
y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred,axis=1)
y_df = pd.DataFrame(data={'ImageId':range(1,28001),'Label':y_pred})
y_df.to_csv("Predictions.csv",index=False)
