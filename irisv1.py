import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

print("Keys of iris dataset={}".format(iris_dataset.keys()))
print("data={}".format(iris_dataset['data']))
print("Target={}".format(iris_dataset['target']))


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

X_new = np.array([[4,4.1,4,0.1]])
print("X_new.shape:{}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))

print("Test set score:{:.2f}".format(np.mean(y_test)))
