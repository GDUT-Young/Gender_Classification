import numpy as np
import pandas as pd
import time
from itertools import combinations
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils import check_array, check_X_y

# VFDT node class
class VfdtNode:
    def __init__(self, possible_split_features):
        """
        nijk: statistics of feature i, value j, class
        possible_split_features: features list
        """
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None  # both continuous and discrete value
        self.new_examples_seen = 0
        self.total_examples_seen = 0
        self.class_frequency = {}
        self.nijk = {f: {} for f in possible_split_features}
        self.possible_split_features = possible_split_features

    def add_children(self, split_feature, split_value, left, right):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left_child = left
        self.right_child = right
        left.parent = self
        right.parent = self

        self.nijk.clear()  # reset stats
        if isinstance(split_value, list):
            left_value = split_value[0]
            right_value = split_value[1]
            # discrete split value list's length = 1, stop splitting
            if len(left_value) <= 1:
                new_features = [None if f == split_feature else f
                                for f in left.possible_split_features]
                left.possible_split_features = new_features
            if len(right_value) <= 1:
                new_features = [None if f == split_feature else f
                                for f in right.possible_split_features]
                right.possible_split_features = new_features

    def is_leaf(self):
        return self.left_child is None and self.right_child is None

    # recursively trace down the tree
    # to distribute data examples to corresponding leaves
    def sort_example(self, x):
        if self.is_leaf():
            return self
        else:
            index = self.possible_split_features.index(self.split_feature)
            value = x[index]
            split_value = self.split_value

            if isinstance(split_value, list):  # discrete value
                if value in split_value[0]:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)
            else:  # continuous value
                if value <= split_value:
                    return self.left_child.sort_example(x)
                else:
                    return self.right_child.sort_example(x)

    # the most frequent class
    def most_frequent(self):
        try:
            prediction = max(self.class_frequency,
                             key=self.class_frequency.get)
        except ValueError:
            # if self.class_frequency dict is empty, go back to parent
            class_frequency = self.parent.class_frequency
            prediction = max(class_frequency, key=class_frequency.get)
        return prediction

    # update leaf stats in order to calculate gini
    def update_stats(self, x, y):
        feats = self.possible_split_features
        nijk = self.nijk
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

    def check_not_splitting(self):
        # compute gini index for not splitting
        X0 = 1
        class_frequency = self.class_frequency
        n = sum(class_frequency.values())
        for j, k in class_frequency.items():
            X0 -= (k/n)**2
        return X0

    # use Hoeffding tree model to test node split, return the split feature
    def attempt_split(self, delta, nmin, tau):
        if self.new_examples_seen < nmin:
            return None
        class_frequency = self.class_frequency
        if len(class_frequency) == 1:
            return None

        self.new_examples_seen = 0  # reset
        nijk = self.nijk
        min = 1
        second_min = 1
        Xa = ''
        split_value = None
        for feature in self.possible_split_features:
            if feature is not None:
                njk = nijk[feature]
                gini, value = self.gini(njk, class_frequency)
                if gini < min:
                    min = gini
                    Xa = feature
                    split_value = value
                elif min < gini < second_min:
                    second_min = gini

        epsilon = self.hoeffding_bound(delta)
        g_X0 = self.check_not_splitting()
        if min < g_X0:
            # print(second_min - min, epsilon)
            if second_min - min > epsilon:
                # print('1 node split')
                return [Xa, split_value]
            elif tau != 0 and second_min - min < epsilon < tau:
                # print('2 node split')
                return [Xa, split_value]
            else:
                return None
        return None

    def hoeffding_bound(self, delta):
        n = self.total_examples_seen
        R = np.log(len(self.class_frequency))
        return np.sqrt(R * R * np.log(1/delta) / (2 * n))

    def gini(self, njk, class_frequency):
        # gini(D) = 1 - Sum(pi^2)
        # gini(D, F=f) = |D1|/|D|*gini(D1) + |D2|/|D|*gini(D2)

        D = self.total_examples_seen
        m1 = 1  # minimum gini
        # m2 = 1  # second minimum gini
        Xa_value = None
        feature_values = list(njk.keys())  # list() is essential
        if not isinstance(feature_values[0], str):  # numeric  feature values
            sort = np.array(sorted(feature_values))
            # vectorized computation, like in R
            split = (sort[0:-1] + sort[1:])/2

            D1_class_frequency = {j: 0 for j in class_frequency.keys()}
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
                        D2_class_frequency[key] = value - \
                            D1_class_frequency[key]
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
            if length > 10:  # too many discrete feature values, estimate
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
                    D1_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    D2_class_frequency = {
                        key: 0 for key in class_frequency.keys()}
                    for j, k in njk.items():
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
                    for key, v in D1_class_frequency.items():
                        g_d1 -= (v/D1)**2
                    for key, v in D2_class_frequency.items():
                        g_d2 -= (v/D2)**2
                    g = g_d1*D1/D + g_d2*D2/D
                    if g < m1:
                        m1 = g
                        Xa_value = left
                    # elif m1 < g < m2:
                        # m2 = g
                right = list(np.setdiff1d(feature_values, Xa_value))
            return [m1, [Xa_value, right]]

    # divide values into two groups, return the combination of left groups
    def select_combinations(self, feature_values):
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


# very fast decision tree class, i.e. hoeffding tree
class Vfdt:
    def __init__(self, features, delta=0.01, nmin=100, tau=0.1):
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
        self.root = VfdtNode(features)
        self.n_examples_processed = 0
        # print(self.features, self.delta, self.tau, self.n_examples_processed)
        # self.print_tree()
        # print("--- / __init__ ---")

    # update the tree by adding one or many training example(s)
    def update(self, X, y):
        X, y = check_X_y(X, y)
        for x, _y in zip(X, y):
            self.__update(x, _y)
        # print("Update!")
        # print("X {}: {}".format(type(X), X))
        # print("y {}: {}".format(type(y), y))
        # self.print_tree()
        # print("---")
        # if isinstance(y, (np.ndarray, list)):
        #     for x, _y in zip(X, y):
        #         self.__update(x, _y)
        # else:
        #     self.__update(X, y)
        # self.print_tree()
        # print("---")
        # print("End update! n_examples_processed={}".format(
        #     self.n_examples_processed))
        # print("--- --- ---")

    # update the tree by adding one training example
    def __update(self, x, _y):
        self.n_examples_processed += 1
        node = self.root.sort_example(x)
        node.update_stats(x, _y)

        result = node.attempt_split(self.delta, self.nmin, self.tau)
        if result is not None:
            feature = result[0]
            value = result[1]
            self.node_split(node, feature, value)

    # split node, produce children
    def node_split(self, node, split_feature, split_value):
        features = node.possible_split_features
        # print('node_split')
        left = VfdtNode(features)
        right = VfdtNode(features)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    def predict(self, X):
        X = check_array(X)
        return [self.__predict(x) for x in X]
        # if isinstance(X, (np.ndarray, list)):
        #     return [self.__predict(x) for x in X]
        # else:
        #     leaf = self.__predict(X)

    def __predict(self, x):
        leaf = self.root.sort_example(x)
        return leaf.most_frequent()

    def print_tree(self, node=None):
        if node is None:
            self.print_tree(self.root)
        elif node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)


def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(
        precision_recall_fscore_support(
            y_test, y_pred, average='weighted',
            labels=np.unique(y_pred)))
    metrics = pd.DataFrame({
        'accuracy': accuracy,
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2]}, index=[row_name])
    return metrics


def test_run():
    start_time = time.time()

    df = pd.read_csv('./dataset/waveform.data', header=None, sep=',')
    # df = pd.read_csv('./dataset/default_of_credit_card_clients.csv', skiprows=1, header=0)
    # df = df.drop(df.columns[0], axis=1)
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data rows
    title = list(df.columns.values)
    features = title[:-1]
    rows = df.shape[0]

    print("Features:", features)

    # change month string to int
    def month_str_to_int(df1):
        import calendar
        d = dict((v.lower(), k) for k, v in enumerate(calendar.month_abbr))
        df1.month = df1.month.map(d)
    # month_str_to_int(df)
    # print(df.head(5)['month'])
    # convert df to data examples
    n_training = 4500
    array = df.head(n_training).values

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
    # tie breaking: when difference is so small,
    # split when diff_g < epsilon < tau
    tree = Vfdt(features, delta=0.01, nmin=100, tau=0.5)
    print('Total dataset size: ', rows)
    print('Training size: ', n_training)
    print('Test set size: ', n_test)
    n = 0
    for training_set in examples:
        n += len(training_set)
        x_train = training_set[:, :-1]
        y_train = training_set[:, -1]
        tree.update(x_train, y_train)
        y_pred = tree.predict(x_test)

        print('Training samples:', n, end=', ')
        print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))

    # tree.print_tree(tree.root)
    print("--- Running time: %.6f seconds ---" % (time.time() - start_time))


# if __name__ == "__main__":
#     test_run()





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

XZY = imgdata          # 图像数据
YZY = imginfor        # 标签
print(type(XZY))
print(type(YZY))
# 特征降维（PCA提取特征）
pca = PCA(n_components=100)
#n_components = 100   # 降维后的特征维数
#imgdata_PCA = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(imgdata)
newX = pca.fit_transform(XZY)

#def main():
dataSet=newX.tolist()
data={}
for a in range(len(dataSet[0])):
	medialist=[]
	for example in range(len(dataSet)):
		medialist.append(dataSet[example][a])
	data[a]=medialist
data['y']=YZY

start_time = time.time()

df = pd.DataFrame(data)
# df = pd.read_csv('./dataset/default_of_credit_card_clients.csv', skiprows=1, header=0)
# df = df.drop(df.columns[0], axis=1)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle data rows
title = list(df.columns.values)
features = title[:-1]
rows = df.shape[0]

print("Features:", features)


# change month string to int
# def month_str_to_int(df1):
#     import calendar
#     d = dict((v.lower(), k) for k, v in enumerate(calendar.month_abbr))
#     df1.month = df1.month.map(d)


# month_str_to_int(df)
# print(df.head(5)['month'])
# convert df to data examples
n_training = 3000   #111111111111111111原先是4500
array = df.head(n_training).values

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
# tie breaking: when difference is so small,
# split when diff_g < epsilon < tau
tree = Vfdt(features, delta=0.01, nmin=100, tau=0.5)
print('Total dataset size: ', rows)
print('Training size: ', n_training)
print('Test set size: ', n_test)
n = 0
for training_set in examples:
    n += len(training_set)
    x_train = training_set[:, :-1]
    y_train = training_set[:, -1]
    tree.update(x_train, y_train)
    y_pred = tree.predict(x_test)

    print('Training samples:', n, end=', ')
    print('ACCURACY: %.4f' % accuracy_score(y_test, y_pred))

# tree.print_tree(tree.root)
print("--- Running time: %.6f seconds ---" % (time.time() - start_time))