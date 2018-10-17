# 其中就有KNN算法
from sklearn import neighbors

from sklearn import datasets

# 调用这个分离器
knn = neighbors.KNeighborsClassifier()
# 调用整个数据
iris = datasets.load_iris()

# print(iris)

# 利用fit函数来建立一个模型
knn.fit(iris.data, iris.target)
# 对一个新的对象进行一个预测
predictedLabel = knn.predict([[0.1,0.2,0.3,0.4]])
print(predictedLabel)

help(knn)