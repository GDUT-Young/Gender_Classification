# coding=utf-8
import numpy as np
import os
import cv2
from sklearn import neighbors, model_selection
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def get_label_from_file(file_path):
    label_num_dict = {'male': 0, 'female': 1}
    labels_np = np.array([])
    with open(file_path) as f:
        for each_line in f:
            begin = each_line.find('_sex ')
            end = each_line.find(')', begin)
            if begin == -1:
                continue
            sex = each_line[begin + 6:end]
            labels_np = np.append(labels_np, label_num_dict[sex])
            # label_index = re.sub(r'\D', "", each_line)
            # print(label_index)
        f.close()
    return labels_np

def create_data_base(path):
    file_list = os.listdir(path)
    data_np = []
    for file in file_list:
        input_file_name = r'C:\Users\Lenovo\Desktop\人脸图像识别\人脸图像识别\face\rawdata'
        input_file_name = input_file_name + '\\' + file
        nx = 128
        ny = 128
        file_load = np.fromfile(input_file_name, dtype='u1')
        if file_load.shape[0] != nx * ny:
            file_load = np.reshape(file_load, (512, 512))
            file_load = cv2.resize(file_load, (128, 128))
        file_load = np.reshape(file_load, (nx * ny,))
        # print(file)
        data_np.append(file_load)
    data_np = np.array(data_np)
    return data_np

def draw_chart(dimension, accuracy, line_color):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(6,6))
    plt.plot(dimension, accuracy, line_color)
    plt.xlabel('PCA降维的维度')
    plt.ylabel('人脸识别准确率')
    plt.title('对比不同核函数训练结果准确性的差异 ')
    plt.savefig("result.png")
    plt.show()

def face_fuc(data_train, labels_train, data_test, label_test, k, c):
    dimension = [5, 10, 20, 50, 100, 200,300]
    accuracy = []
    for i in dimension:
        pca = PCA(n_components=i).fit(data_train)
        data_numpy_train_pca = pca.transform(data_train)
        data_numpy_test_pca = pca.transform(data_test)

        # data standard
        stdScaler = StandardScaler().fit(data_numpy_train_pca)
        data_numpy_train_pca = stdScaler.transform(data_numpy_train_pca)
        data_numpy_test_pca = stdScaler.transform(data_numpy_test_pca)

        svm = SVC(kernel=k).fit(data_numpy_train_pca, labels_train)
        face_target_pred = svm.predict(data_numpy_test_pca)
        true = 0
        for i in range(0,len(face_target_pred)):
            if face_target_pred[i] == label_test[i]:
                true += 1
        accuracy.append(true / label_test.shape[0])
    print(accuracy)
    plt.plot(dimension, accuracy, c)

# get data use pca
data_path = r'C:\Users\Lenovo\Desktop\人脸图像识别\人脸图像识别\face\rawdata'
data_numpy = create_data_base(data_path)

# get label
file_name_DR = r'C:\Users\Lenovo\Desktop\人脸图像识别\人脸图像识别\face\faceDR'
file_name_DS = r'C:\Users\Lenovo\Desktop\人脸图像识别\人脸图像识别\face\faceDS'
DR_numpy = get_label_from_file(file_name_DR)
DS_numpy = get_label_from_file(file_name_DS)
labels_numpy = np.append(DR_numpy, DS_numpy)

# get the min class num
each_class_num = min(len(data_numpy[labels_numpy == 0]), len(data_numpy[labels_numpy == 1]))

# quantity_proportion
quantity_proportion = [0.9, 0.1]
# get dataset
train_index = int(quantity_proportion[0]*each_class_num)

data_numpy_train = np.vstack((data_numpy[labels_numpy == 0][0:train_index],
                              data_numpy[labels_numpy == 1][0:train_index]))
data_numpy_test = np.vstack((data_numpy[labels_numpy == 0][train_index:each_class_num],
                             data_numpy[labels_numpy == 1][train_index:each_class_num]))
labels_numpy_train = np.hstack((labels_numpy[labels_numpy == 0][0:train_index],
                                labels_numpy[labels_numpy == 1][0:train_index]))
labels_numpy_test = np.hstack((labels_numpy[labels_numpy == 0][train_index:each_class_num],
                               labels_numpy[labels_numpy == 1][train_index:each_class_num]))

# data standard
stdScaler = StandardScaler().fit(data_numpy_train)
data_numpy_train = stdScaler.transform(data_numpy_train)
data_numpy_test = stdScaler.transform(data_numpy_test)

# data_numpy_pca = data_numpy
#kernel = ['poly', 'rbf']
#color = ['-r', '-y']
kernel = ['poly']
color = ['-r']

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize=(6, 6))
plt.xlabel('Dimension')
plt.ylabel('Recognition rate')
plt.title('The accuracy difference')
for i in range(0, 1):
    face_fuc(data_numpy_train, labels_numpy_train, data_numpy_test, labels_numpy_test, kernel[i], color[i])
plt.legend(['poly'])
plt.savefig("result.png")
plt.show()

