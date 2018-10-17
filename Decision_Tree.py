from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import  StringIO

allEkectronicsData = open(r'F:\cycle ingenieur 2\semestre9\EXERCICES\Notes_of_machine_learning\first.csv',"r")
reader = csv.reader(allEkectronicsData)
headers = next(reader)

# print(headers)

featureList = []
labelList = []

# 创建一个字典以便之后的操作
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row [i]
    featureList.append(rowDict)

# print(featureList)
# print(labelList)

vec = DictVectorizer()
# 转化为特征值的数字化模式
dummyX = vec.fit_transform(featureList).toarray()

# print("dummy : X" + str(dummyX))
# 查看每个项目路对应的含义
# print(vec.get_feature_names())

# print("label list: " + str(labelList))

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
# print("dummyY : " + str (dummyY))

# 声明创造分离器
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(dummyX, dummyY)
# 查看决策树的参数
# print("clf:" +str(clf))

with open("allElE.dot","w") as f:
    # feature_names = vec.get_feature_names()把featurenames改回去
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)


# 预测环节
oneRowX = dummyX[0 , :]
# print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print(newRowX)


newRowX.reshape(-1,1)


# 这里必须将这份函数2D化
prediction = clf.predict([newRowX])

print(prediction)