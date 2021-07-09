import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

os.chdir(r"E:\桌面\大三\模式识别\课程项目\FR\rawdata") #定位到图像文件位置
file_list = os.listdir("./")                         #file_list为遍历图像文件名称 list类型
rows = 128   #行数
cols = 128   #列数
channal = 1  #通道数，灰度图为1
imgdata = np.empty((1,16384),dtype='uint8') #创建空数组存放图像信息

# 遍历图像信息存入imgdata中
# imgdata为 3991*16384 维的数组（除去'2416'和'2412'（图像大小存在错误，暂不考虑））
for file_name in file_list:
    img = np.fromfile(file_name, dtype='uint8')   
    img = img.reshape(1,16384)
    print(file_name)
    imgdata = np.vstack((imgdata,img))
imgdata = np.delete(imgdata,0,axis=0)        #删除第一行；axis=0，为删除行；axis=1为删除列
print("Successfully Import The Face Data in imgdata!!")

# 类别
f = open(r"E:\桌面\大三\模式识别\课程项目\FR\face.txt")
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
# 删除缺失图像信息的信息
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
      % (len(imginfor),imginfor.count('male'),imginfor.count('female'))) 
print("===================================================")
f.close()

X = imgdata          # 图像数据
Y = imginfor         # 标签

# PCA
n_component = 100   # 降维后的特征维数
bayes_PCA = PCA(n_components=n_component, svd_solver='randomized', whiten=True).fit(X)
x_pca = bayes_PCA.transform(X)       
y_pca = Y
print("------Naive Bayes with PCA------")
print("The size of PCA characteristic matrix:")
print(x_pca.shape)
X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(x_pca,y_pca, test_size=0.3, random_state=2) 
print("The shape of training and of testing : ")
print(X_train_PCA.shape, X_test_PCA.shape)
clf_bayes_pca = GaussianNB()
y_pred_bayes_PCA = clf_bayes_pca.fit(X_train_PCA, y_train_PCA).predict(X_test_PCA)
print("The accuracy of Bayes with pca is :",accuracy_score(y_test_PCA, y_pred_bayes_PCA))
print("The confusion matricx of Bayes with PCA is :")
print(confusion_matrix(y_test_PCA, y_pred_bayes_PCA))
cross validation
gnb_cross = GaussianNB()
for i in range(10,200,10):
     scores = cross_val_score(gnb_cross, x_pca, y_pca, cv=i, scoring='accuracy')
     print("Bayes(cross-validation) accuracy:%0.3f " % (scores.mean()))
print("============================================================")

# LDA
x_lda = imgdata
y_lda = imginfor
X_train_LDA, X_test_LDA, y_train_LDA, y_test_LDA = train_test_split(x_lda,y_lda, test_size=0.3, random_state=2) # 划分训练集
LDA = LinearDiscriminantAnalysis(n_components = 1).fit(X_train_LDA,y_train_LDA)
x_train_lda = LDA.transform(X_train_LDA)
x_test_lda = LDA.transform(X_test_LDA)
print("------Naive Bayes with LDA------")
print("The shape of training and of testing : ")
print(X_train_LDA.shape, X_test_LDA.shape)
clf_bayes_lda = GaussianNB()
y_pred_bayes_LDA = clf_bayes_lda.fit(X_train_LDA, y_train_LDA).predict(X_test_LDA)
print("The accuracy of Bayes with LDA is :",accuracy_score(y_test_LDA, y_pred_bayes_LDA))
print("The confusion matricx of Bayes with LDA is :")
print(confusion_matrix(y_test_LDA, y_pred_bayes_LDA))
## cross validation
gnb_cross = GaussianNB()
x_cross_lda = LDA.transform(X)
scores = cross_val_score(gnb_cross, x_cross_lda, Y, cv=50, scoring='accuracy')
print("Bayes(50 cross-validation) with LDA accuracy:%0.3f " % (scores.mean()))
print("============================================================")

# PCA+LDA
PCA_LDA = LinearDiscriminantAnalysis(n_components = 1).fit(X_train_PCA,y_train_PCA)
x_train_pl = PCA_LDA.transform(X_train_PCA)
x_test_pl = PCA_LDA.transform(X_test_PCA)
clf_bayes_pl = GaussianNB()
y_pred_bayes_PL = clf_bayes_lda.fit(x_train_pl, y_train_PCA).predict(x_test_pl)
print("------Naive Bayes with PCA-LDA------")
print("The accuracy of Bayes with PCA-LDA is :",accuracy_score(y_test_PCA, y_pred_bayes_PL))
print("The confusion matricx of Bayes with PCA-LDA is :")
print(confusion_matrix(y_test_PCA, y_pred_bayes_PL))
## cross validation
cross_pl = LinearDiscriminantAnalysis(n_components = 1).fit(x_pca,y_pca)
x_pl = cross_pl.transform(x_pca)
gnb_cross = GaussianNB()
scores = cross_val_score(gnb_cross, x_pl, Y, cv=50, scoring='accuracy')
print("Bayes(50 cross-validation)with PCA and LDA accuracy:%0.3f " % (scores.mean()))
print("============================================================")

# Influence of PCA dimension
acc1 = []
X = imgdata         
Y = imginfor     
for i in range(100,1000,100):
    n_component = i   
    bayes_PCA = PCA(n_components = n_component, svd_solver='randomized', whiten=True).fit(X)
    x_pca = bayes_PCA.transform(X)      
    y_pca = Y
    X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(x_pca,y_pca, test_size=0.3, random_state=2) # 划分训练集
    clf_bayes_pca = GaussianNB()
    y_pred_bayes_PCA = clf_bayes_pca.fit(X_train_PCA, y_train_PCA).predict(X_test_PCA)
    acc1.append(accuracy_score(y_test_PCA, y_pred_bayes_PCA))
plt.plot(range(100,1000,100),acc1)
plt.xlabel('PCA Dimension')
plt.ylabel('Accuracy')
plt.title('The accuracy of different PCA dimension of Gender classification')
