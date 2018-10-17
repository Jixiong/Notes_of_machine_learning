print()

import pylab as pl
import numpy as np
from  sklearn import svm

np.random.seed(0)
# randn 正态分布
X = np.r_[np.random.randn(20,2) - [2,2], np.random.randn(20,2) + [2,2]]
y = [0] * 20 + [1] * 20

clf = svm.SVC( kernel = 'linear' )
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]/w[1])

b = clf.support_vectors_[0]
yy_dowm = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print(w)
print(a)
print(clf.support_vectors_)
print(clf.coef_)

pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_dowm, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=80, facecolor = 'none')
pl.scatter(X[:,0], X[:,1], c = y, cmap = pl.cm.Paired)

pl.axis('tight')
pl.show()