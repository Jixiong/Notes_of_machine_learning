from  sklearn import svm

X = [[1,1], [2,0], [2,3]]
# y is the class of the X
y = [0,0,1]

# clf classical file
clf = svm.SVC( kernel = 'linear' )
clf.fit(X, y)

print(clf)

# get support vectors
print(clf.support_vectors_)

# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)

print(clf.predict([[11,15]]))