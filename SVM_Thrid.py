from __future__ import print_function

from time import time
import logging                  #打印程序的进展
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np

from PIL import Image

print(__doc__)



logging.basicConfig(level=logging.INFO, format='%(asctime)s%(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

Y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_features: %d" % n_classes)

X_train, X_test, y_train, y_text = train_test_split(X, Y, test_size = 0.25 )

n_components = 150

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components = n_components, whiten = True).fit(X_train)
print("done in %0.3fs" %(time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)
print("done in %0.3fs"%(time()-t0))

print("Fitting the classifier to the training set")
to = time()
param_grid = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
}



clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_test_PCA.reshape((966,322)), y_train)
print("Done in %0.3fs" %(time() - t0))
print("Best estimator found by frid search")
print(clf.best_estimator_)

print("Predicting people's names on the stest set")
to = time()
y_pred = clf.predict(X_test_PCA)
print("Done in %0.3fs" %(time() - t0))

print(classification_report(y_text, y_pred, target_names = target_names))
print(confusion_matrix(y_text, y_pred, labels = range(n_classes)))

# def plot_gallery(images, titles, h, w, n_row =3, n_col = 4):