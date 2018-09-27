import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

print("Keys of iris dataset:{}".format(iris_dataset.keys()))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

iris_dataset = pd.DataFrame(X_train,columns=iris_dataset.feature_names)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

X_new = np.array([[4,4.1,4,0.1]])
print("X_new.shape:{}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))

y_pred = knn.predict(X_testṭ)                                                                                            #y=f(x)
print("Test set predictions:\n{}".format(y_pred))

print("Test set score:{:.2f}".format(np.mean(y_pred == y_test)))


















