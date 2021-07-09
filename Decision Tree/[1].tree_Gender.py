import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import _base
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

os.chdir(r"D:\学校课程\大三下\模式识别\人脸图像识别人脸样本集\face\rawdata") #定位到图像文件位置
file_list = os.listdir("./")                        #file_list为遍历图像文件名称 list类型
rows = 128   #行数
cols = 128   #列数
channal = 1  #通道数，灰度图为1
imgdata = np.empty((1,16384),dtype='uint8') #创建空数组存放图像信息

# 遍历图像信息存入imgdata中
for file_name in file_list:
	img = np.fromfile(file_name, dtype='uint8')
	img = img.reshape(1,16384)
	print(file_name)
	imgdata = np.vstack((imgdata,img))
imgdata = np.delete(imgdata,0,axis=0)
print("Successfully Import The Face Data in imgdata!!")
f = open(r"D:\学校课程\大三下\模式识别\人脸图像识别人脸样本集\face\FR.txt")
lines = f.readlines()
img_information = []
imginfor = []
wrong = []
for line in lines:
	temp1 = line.strip('\n')    # strip()为删除函数；删去空行
	temp3 = temp1.split('(')    # 以 “（” 分割
	img_information.append(temp3)
for i in range(len(img_information)):
	if img_information[i][1].strip() != '_missing descriptor)':
		sex = img_information[i][1].strip()
		gesture = img_information[i][4].strip()
		age = img_information[i][2].strip()
		if sex == '_sex  male)':
			imginfor.append('male')
		else:
			imginfor.append('female')
	else:
		imginfor.append('missing')
		wrong.append(i)
index_offset = 0
for i in range(len(wrong)):
	a = wrong[i] - index_offset
	del img_information[a]
	del imginfor[a]
	index_offset += 1
offset = 0
for j in wrong:
	if j < 1189 and j > 0:
		offset += 1
del imginfor[1189-offset]
del imginfor[1193-offset-1]
del img_information[1189-offset]
del img_information[1193-offset-1]
print("Successfully Import The Face Label In imginfo!!")
print("Total face image:%d\nThe number of male:%d\nThe number of female:%d"
	  % (len(imginfor),imginfor.count('male'),imginfor.count('female')))  #类别样本数 2425 1566
print("===================================================")
f.close()

X = imgdata          # 图像数据
Y = imginfor        # 标签

# 特征降维（PCA提取特征）
pca = PCA(n_components=100)
newX = pca.fit_transform(X)

dataSet=newX.tolist()
dataSetset={'data':dataSet,'target':Y}
num=0

from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = dataSetset
X = dataSet
y = Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1, stratify = y )

#训练
clf = tree.DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

#测试
score = clf.score(X_test,y_test)
print('预测准确率：',score)


