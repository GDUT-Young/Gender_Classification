import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

trainningnum = 2000
os.chdir("rawdata") #定位到图像文件位置
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

f = open(r"C:\Users\戴尔\Desktop\Minimum_distance_classification-master/face.txt")
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
n_components = 100   # 降维后的特征维数
pca = PCA(n_components=n_components)
newX = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)

Accuracy =[]
males = []
females = []
length = min(trainningnum,len(newX))
for i in range(length):
    if Y[i] =='male':
        males.append(list(newX[i]))
    else:
        females.append(list(newX[i]))

males = np.array(males).reshape(-1,n_components)
females = np.array(females).reshape(-1,n_components)
###########################求出1和0的训练样本的特征中心########################

males_center = np.sum(males,axis=0)/len(males)
males_d= (males-males_center)**2
males_d = (np.sum(males_d,axis = 0)/len(males_d))**(1/2)

females_center = np.sum(females,axis=0)/len(females)
females_d= (females-females_center)**2
females_d = (np.sum(females_d,axis = 0)/len(females_d))**(1/2)

########################最小分类算法########################################
def mindistance(textsample):
    males_center_t = males_center.reshape(1, -1).repeat(len(textsample), axis=0)
    females_center_t = females_center.reshape(1, -1).repeat(len(textsample), axis=0)
    males_d_t = males_d.reshape(1, -1).repeat(len(textsample), axis=0)
    females_d_t = females_d.reshape(1, -1).repeat(len(textsample), axis=0)
    distance1 = (textsample-males_center_t)**2
    distance0 = (textsample-females_center_t)**2
    distance1 = np.sum(distance1,axis =1)/n_components
    distance0 = np.sum(distance0,axis=1)/n_components
    return distance0,distance1

#对datas中的数据进行求解，首先先把其归一化
datas = newX
distance0,distance1 = mindistance(datas)
Yout = []
for i in range(len(datas)):
    if distance0[i]>distance1[i]:
        result = 'male'
    else:
        result = 'female'
    Yout.append(result)
print("The result of prediction")
print(Yout)

js =0#计数，如果预测结果和真实相等就加1
for i in range(0, len(Yout)):
    if Yout[i] == Y[i]:
        js += 1
print("Total face image:%d\nThe number of male:%d\nThe number of female:%d"
      % (len(datas),Yout.count('male'),Yout.count('female')))
print("Accuracy:%f\n"
      % (float(js/len(Yout))))
print("===================================================")

# x = np.arange(minnum, maxnum, 1)
# y = np.array(Accuracy)
# plt.plot(x,y,label='Training data and Accuracy', linewidth=2)
# plt.xlabel('Training data') # 横坐标轴的标题
# plt.ylabel('Accuracy') # 纵坐标轴的标题
# plt.legend()
# plt.title('Training data and Accuracy')
#
# plt.show()