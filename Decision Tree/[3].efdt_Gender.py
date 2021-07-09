import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import combinations
from sklearn.metrics import accuracy_score

# EFDT node class
class EfdtNode:
    def __init__(self, possible_split_features):
        """
        nijk: statistics of feature i, value j, class
        possible_split_features: features list
        """
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.split_g = None
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {i: {} for i in possible_split_features}
        self.possible_split_features = possible_split_features

    def add_children(self, left, right, split_feature, split_value):
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self
        self.nijk = {i: {} for i in self.nijk.keys()}

        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # update node stats in order to calculate Gini
    def update_stats(self, x, y):
        nijk = self.nijk
        feats = self.possible_split_features
        iterator = [f for f in feats if f is not None]
        for i in iterator:
            value = x[feats.index(i)]

            if value not in nijk[i]:
                nijk[i][value] = {y: 1}
            else:
                try:
                    nijk[i][value][y] += 1
                except KeyError:
                    nijk[i][value][y] = 1

        self.total_examples_seen += 1
        self.new_examples_seen += 1
        class_frequency = self.class_frequency
        try:
            class_frequency[y] += 1
        except KeyError:
            class_frequency[y] = 1

    # the most frequent classification
    def most_frequent(self):
        if self.class_frequency:
            prediction = max(self.class_frequency, key=self.class_frequency.get)
        else:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    def check_not_splitting(self, class_frequency):
        # compute Gini index for not splitting
        g_d1 = 1
        # g_d2 = 1
        # most = max(class_frequency, key=class_frequency.get)
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            g_d1 -= (k/n)**2
        return g_d1

    def node_split(self, split_feature, split_value):
        self.split_feature = split_feature
        self.split_value = split_value
        features = self.possible_split_features

        left = EfdtNode(features)
        right = EfdtNode(features)
        self.add_children(left, right, split_feature, split_value)

    # recursively trace down the tree
    def sort_example(self, x, y, delta, nmin, tau):
        self.update_stats(x, y)
        self.re_evaluate_split(delta, nmin, tau)
        if self.is_leaf():
            self.attempt_split(delta, nmin, tau)
            return
        else:
            left = self.left_child
            right = self.right_child

            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return left.sort_example(x, y, delta, nmin, tau)
                else:
                    return right.sort_example(x, y, delta, nmin, tau)
            else:  # continuous value
                if value <= split_value:
                    return left.sort_example(x, y, delta, nmin, tau)
                else:
                    return right.sort_example(x, y, delta, nmin, tau)

    def sort_to_predict(self, x):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            split_value = self.split_value
            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_to_predict(x)
                else:
                    return self.right_child.sort_to_predict(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_to_predict(x)
                else:
                    return self.right_child.sort_to_predict(x)

    # test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_examples_seen < nmin:
            return
        class_frequency = self.class_frequency
        if len(class_frequency) <= 1:
            return

        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        g_Xa = 1  # minimum g

        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                gini, value = self.gini(njk, class_frequency)
                if gini < g_Xa:
                    g_Xa = gini
                    Xa = feature
                    split_value = value

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting(class_frequency)
        g_Xb = g_X0
        if g_Xa < g_X0:
            if g_Xb - g_Xa > epsilon:
                self.split_g = g_Xa  # split on feature Xa
                # print('1 node split')
                self.node_split(Xa, split_value)
            elif g_Xb - g_Xa < epsilon < tau:
                self.split_g = g_Xa  # split on feature Xa
                # print('2 node split')
                self.node_split(Xa, split_value)

    def re_evaluate_split(self, delta, nmin, tau):
        if self.new_examples_seen < nmin or self.is_leaf():  # only re-evaluate non-leaf
            return
        class_frequency = self.class_frequency
        if len(class_frequency) <= 1:
            return

        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        g_Xa = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                gini, value = self.gini(njk, class_frequency)
                if gini < g_Xa:
                    g_Xa = gini
                    Xa = feature
                    split_value = value

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting(class_frequency)
        split_g = self.split_g  # gini of current split feature

        if g_X0 < g_Xa:  # not split
            print('kill subtree')
            self.kill_subtree()
        if split_g - g_Xa > epsilon or split_g - g_Xa < epsilon < tau:
            if Xa != self.split_feature:
                # print('split on new feature')
                self.split_g = g_Xa  # split on feature Xa
                self.node_split(Xa, split_value)

    def kill_subtree(self):
        if not self.is_leaf():
            self.left_child = None
            self.right_child = None
            self.split_feature = None
            self.split_value = None
            self.split_g = None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return (R * R * np.log(1/delta) / (2 * n))**0.5

    def gini(self, njk, class_frequency):
        # Gini(D) = 1 - Sum(pi^2)
        # Gini(D, F=f) = |D1|/|D|*Gini(D1) + |D2|/|D|*Gini(D2)
        D = self.total_examples_seen
        m1 = 1  # minimum gini
        # m2 = 1  # second minimum gini
        Xa_value = None
        feature_values = list(njk.keys())  # list() is essential
        if not isinstance(feature_values[0], str):  # numeric feature values
            sort = np.array(sorted(feature_values))
            split = (sort[0:-1] + sort[1:])/2   # vectorized computation, like in R

            D1_class_frequency = {j:0 for j in class_frequency.keys()}
            for index in range(len(split)):
                nk = njk[sort[index]]

                for j in nk:
                    D1_class_frequency[j] += nk[j]
                D1 = sum(D1_class_frequency.values())
                D2 = D - D1
                g_d1 = 1
                g_d2 = 1

                D2_class_frequency = {}
                for key, value in class_frequency.items():
                    if key in D1_class_frequency:
                        D2_class_frequency[key] = value - D1_class_frequency[key]
                    else:
                        D2_class_frequency[key] = value

                for key, v in D1_class_frequency.items():
                    g_d1 -= (v/D1)**2
                for key, v in D2_class_frequency.items():
                    g_d2 -= (v/D2)**2
                g = g_d1*D1/D + g_d2*D2/D
                if g < m1:
                    m1 = g
                    Xa_value = split[index]
                # elif m1 < g < m2:
                    # m2 = g

            return [m1, Xa_value]

        else:  # discrete feature_values
            length = len(njk)
            if length > 9:  # too many discrete feature values, estimate
                for j, k in njk.items():
                    D1 = sum(k.values())
                    D2 = D - D1
                    g_d1 = 1
                    g_d2 = 1

                    D2_class_frequency = {}
                    for key, value in class_frequency.items():
                        if key in k:
                            D2_class_frequency[key] = value - k[key]
                        else:
                            D2_class_frequency[key] = value
                    for key, v in k.items():
                        g_d1 -= (v/D1)**2

                    if D2 != 0:
                        for key, v in D2_class_frequency.items():
                            g_d2 -= (v/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D

                    if g < m1:
                        m1 = g
                        Xa_value = j
                    # elif m1 < g < m2:
                        # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))

            else:  # fewer discrete feature values, get combinations
                comb = self.select_combinations(feature_values)
                for i in comb:
                    left = list(i)
                    D1_class_frequency = {key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {key: 0 for key in class_frequency.keys()}
                    for j,k in njk.items():
                        for key, value in class_frequency.items():
                            if j in left:
                                if key in k:
                                    D1_class_frequency[key] += k[key]
                            else:
                                if key in k:
                                    D2_class_frequency[key] += k[key]
                    g_d1 = 1
                    g_d2 = 1
                    D1 = sum(D1_class_frequency.values())
                    D2 = D - D1
                    for key1, v1 in D1_class_frequency.items():
                        g_d1 -= (v1/D1)**2
                    for key2, v2 in D2_class_frequency.items():
                        g_d2 -= (v2/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D

                    if g < m1:
                        m1 = g
                        Xa_value = left
                    # elif m1 < g < m2:
                        # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    # divide values into two groups, return the combination of left groups
    @staticmethod
    def select_combinations(feature_values):
        combination = []
        e = len(feature_values)
        if e % 2 == 0:
            end = int(e/2)
            for i in range(1, end+1):
                if i == end:
                    cmb = list(combinations(feature_values, i))
                    enough = int(len(cmb)/2)
                    combination.extend(cmb[:enough])
                else:
                    combination.extend(combinations(feature_values, i))
        else:
            end = int((e-1)/2)
            for i in range(1, end+1):
                combination.extend(combinations(feature_values, i))
        return combination


class Efdt:
    def __init__(self, features, delta=0.05, nmin=50, tau=0.1):
        """
        :features: list of data features
        :delta: used to compute hoeffding bound, error rate
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        self.root = EfdtNode(features)
        self.n_examples_processed = 0

    # find the path of example
    def find_path(self, leaf):
        path = []
        node = leaf
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    # update the tree by adding training example
    def update(self, x, y):
        self.n_examples_processed += 1
        self.root.sort_example(x, y, self.delta, self.nmin, self.tau)

    # predict test example's classification
    def predict(self, x_test):
        prediction = []
        if isinstance(x_test, np.ndarray) or isinstance(x_test, list):
            for x in x_test:
                leaf = self.root.sort_to_predict(x)
                prediction.append(leaf.most_frequent())
            return prediction
        else:
            leaf = self.root.sort_to_predict(x_test)
            return leaf.most_frequent()

    def print_tree(self, node):
        if node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)


def test_run():
    start_time = time.time()
    # bank.csv whole data size: 4521 # skiprows=1, nrows=n
    df = pd.read_csv('./dataset/bank.csv', header=0, sep=';')
    print(df)
    # df = pd.read_csv('./dataset/default_of_credit_card_clients.csv', skiprows=1, header=0)
    # df = df.drop(df.columns[0], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data rows
    print(type(df))
    print(type(df))
    print(df)
    title = list(df.columns.values)
    print(title)
    features = title[:-1]
    print(features)
    rows = df.shape[0]
    ''' change month string to int '''
    # def month_str_to_int(df1):
    #     import calendar
    #     d = dict((v.lower(),k) for k,v in enumerate(calendar.month_abbr))
    #     df1.month = df1.month.map(d)
    # month_str_to_int(df)

    # convert df to data examples
    training_size = 4000
    array = df.head(training_size).values
    set1 = array[:1000, :]
    set2 = array[1000:2000, :]
    set3 = array[2000:, :]

    # to simulate continuous training, modify the tree for each training set
    examples = [set1, set2, set3]

    # test set is different from training set
    n_test = 500
    test_set = df.tail(n_test).values
    x_test = test_set[:, :-1]
    y_test = test_set[:, -1]

    # Heoffding bound (epsilon) parameter delta: with 1 - delta probability
    # the true mean is at least r_bar - epsilon
    # Efdt parameter nmin: test split if new sample size > nmin
    # feature_values: unique values in every feature
    # tie breaking: when difference is so small, split when diff_g < epsilon < tau
    tree = Efdt(features, delta=0.01, nmin=100, tau=0.5)
    print('Total data size: ', rows)
    print('Training size: ', training_size)
    print('Test set size: ', n_test)
    n = 0
    for training_set in examples:
        n += len(training_set)
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        for x, y in zip(x_train, y_train):
            tree.update(x, y)
        y_pred = tree.predict(x_test)
        print('Training samples:', n, end=', ')
        print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))

    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))



# if __name__ == "__main__":
#      test_run()
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
print(type(X))
print(type(Y))
# 特征降维（PCA提取特征）
pca = PCA(n_components=100)
#n_components = 100   # 降维后的特征维数
#imgdata_PCA = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(imgdata)
newX = pca.fit_transform(X)
#def main():
dataSet=newX.tolist()
data={}
for a in range(len(dataSet[0])):
	medialist=[]
	for example in range(len(dataSet)):
		medialist.append(dataSet[example][a])
	data[a]=medialist
data['y']=Y

df = pd.DataFrame(data)
start_time = time.time()
# bank.csv whole data size: 4521 # skiprows=1, nrows=n


#df = pd.read_csv('./dataset/bank.csv', header=0, sep=';')
print(df)
# df = pd.read_csv('./dataset/default_of_credit_card_clients.csv', skiprows=1, header=0)
# df = df.drop(df.columns[0], axis=1)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle data rows
print(type(df))
print('44444444444444444444444444444444444444444444444444444444444\n')
print(type(df))
print(df)
title = list(df.columns.values)
print(title)
features = title[:-1]
print(features)
rows = df.shape[0]
''' change month string to int '''


# def month_str_to_int(df1):
#     import calendar
#     d = dict((v.lower(), k) for k, v in enumerate(calendar.month_abbr))
#     df1.month = df1.month.map(d)
#
#
# month_str_to_int(df)

# convert df to data examples
training_size = 3000
array = df.head(training_size).values
set1 = array[:1000, :]
set2 = array[1000:2000, :]
set3 = array[2000:, :]

# to simulate continuous training, modify the tree for each training set
examples = [set1, set2, set3]

# test set is different from training set
n_test = 500
test_set = df.tail(n_test).values
x_test = test_set[:, :-1]
y_test = test_set[:, -1]

# Heoffding bound (epsilon) parameter delta: with 1 - delta probability
# the true mean is at least r_bar - epsilon
# Efdt parameter nmin: test split if new sample size > nmin
# feature_values: unique values in every feature
# tie breaking: when difference is so small, split when diff_g < epsilon < tau
tree = Efdt(features, delta=0.01, nmin=100, tau=0.5)
print('Total data size: ', rows)
print('Training size: ', training_size)
print('Test set size: ', n_test)
n = 0
for training_set in examples:
    n += len(training_set)
    x_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    for x, y in zip(x_train, y_train):
        tree.update(x, y)
    y_pred = tree.predict(x_test)
    print('Training samples:', n, end=', ')
    print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))

print("--- Running time: %.6f seconds ---" % (time.time() - start_time))