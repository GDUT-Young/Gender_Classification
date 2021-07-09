import os,glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import _base
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import pickle

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)



def data_generate(data_dir,label_txt,saved='data.pkl',force=False):
    # 如果已经保存过，直接读取就好了，不用重复生成
    if not force and os.path.exists(saved):
        return load_obj(saved)


    file_list = glob.glob(data_dir)                        #file_list为遍历图像文件名称 list类型
    imgdata = np.empty((1,128*128*1),dtype='uint8') #创建空数组存放图像信息

    # 遍历图像信息存入imgdata中     imgdata为 3991*16384 维的数组（除去'2416'和'2412'（图像大小存在错误，暂不考虑））
    for file_name in file_list:
        img = np.fromfile(file_name, dtype='uint8')   #img为numpy.ndarray(数组)类型
        img = img.reshape(1,16384)
        print(file_name)
        imgdata = np.vstack((imgdata,img))
    imgdata = np.delete(imgdata,0,axis=0)        #删除第一行；axis=0，为删除行；axis=1为删除列

    # 特征降维（PCA提取特征）
    n_components = 150   # 降维后的特征维数
    print("Extracting the top %d eigenfaces from %d faces" % (n_components, imgdata.shape[1]))
    imgdata_PCA = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(imgdata)
    eigenfaces = imgdata_PCA.components_.reshape((n_components,128,128))
    imgdata_pca = imgdata_PCA.transform(imgdata)       # imgdata_pca为降维后的图像信息，3991 * 200
    print("The size of PCA characteristic matrix:")
    print(imgdata_pca.shape)

    # 读取人脸标签
    f = open(label_txt)
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
    # 暂时删除缺失图像信息的信息
    index_offset = 0          #索引的偏移量：因为下面是逐次删除，所以每当前一次删除后，后面的索引响应减一
    for i in range(len(wrong)):
        a = wrong[i] - index_offset
        del img_information[a]
        del imginfor[a]
        index_offset += 1
    offset = 0
    for j in wrong:
        if j < 1189 and j > 0:
            offset += 1
    del imginfor[1189-offset]          # 删除维度错误的图像  1189 smiling;1193 serious
    del imginfor[1193-offset-1]
    del img_information[1189-offset]
    del img_information[1193-offset-1]
    print("========================================")
    print("Total date:%d,\nmale:%d;\nfemale:%d."
        % (len(imginfor),imginfor.count('male'),imginfor.count('female')))  #类别样本数 2425 1566
    print("========================================")
    f.close()
    label = []
    for name in imginfor:
        if name == 'male':
            label.append(0)
        else:
            label.append(1)
    result = {'feature':imgdata_pca, 'label':label, 'label_name':["male","female"]}
    save_obj(result,'data.pkl')

    return result
  

if __name__ =='__main__':
    data = data_generate("./face/rawdata/*","./face/face1.txt",force=True)
    print(data['feature'].shape)
    print(len(data['label']))
    print(data['label_name'])