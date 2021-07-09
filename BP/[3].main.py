import torch,copy,argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
from data_generater import data_generate
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from ga import *

# 固定随机种子
torch.manual_seed(15)
torch.cuda.manual_seed_all(15)
np.random.seed(15)
torch.backends.cudnn.deterministic = True



def cal_score(model,data,label):
    """ 计算准确率 """
    y_pred = torch.max(F.softmax(model(data),dim=1), 1)[1].numpy()
    accuracy = (y_pred==label).sum()/len(label)
    return accuracy



def main(mode):
    ## ========================数据处理======================
    # 加载数据并预处理
    data = data_generate("./face/rawdata/*","./face/face1.txt")
    feature = data['feature']            # 降维后的特征
    label = data["label"]                # 标签
    # 训练集/验证集/测试集划分
    x_train, x_val_test, y_train, y_val_test = train_test_split(feature,label, test_size=0.4 , random_state=2) # 划分训练集
    x_val, x_test, y_val, y_test = train_test_split(x_val_test,y_val_test, test_size=0.5 , random_state=2) # 划分验证集、测试集
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(np.array(y_train)).type(torch.LongTensor)
    x_val = torch.from_numpy(x_val).type(torch.FloatTensor)
    y_val = torch.from_numpy(np.array(y_val)).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(np.array(y_test)).type(torch.LongTensor)


    ## ========================搭建BP网络======================
    net=torch.nn.Sequential(
        torch.nn.Linear(150,20),   # 第一层输入2个数据，输出10个数据（神经元10个）
        torch.nn.ReLU(),   # 激活函数
        torch.nn.Linear(20,5), 
        torch.nn.ReLU(),   # 激活函数
        torch.nn.Linear(5,2),   
    )
    def train_model(model,x,y, num_epoches, learning_rate=1e-3):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_recorder = []
        for e in range(num_epoches):
            y_pred = model(x).squeeze(-1)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_recorder.append(loss.item())
        return loss



    ## ========================GA初始化BP参数======================
    if mode == 'gabp':
        def calculate_fitness(chrom,model,train_epoches=200):
            """ 用于计算种群适应度的函数 """
            params = chrom_to_param(chrom, param_template)
            model.load_state_dict(params)
            loss = train_model(model,x_val,y_val,num_epoches=train_epoches)
            return 1./loss.item()
        param_template, chrom_len, bound = model_to_template(net)
        pop_size = 20
        ga = GA(pop_size, chrom_len, bound, calculate_fitness, copy.deepcopy(net), cross_prob=0.5, mutate_prob=0.1)
        ga.genetic(10, log=False)
        # 将种群中适应度最高的初始权重赋给模型
        best_ga_param = chrom_to_param(ga.result(), param_template)
        net.load_state_dict(best_ga_param)


    ## ========================BP网络训练======================
    train_model(net,x_train,y_train,num_epoches=200)

    ## ========================计算准确率======================
    accuracy = cal_score(net,x_test,y_test.numpy())
    print('accuracy =',accuracy)
    return accuracy

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--mode', type=str, default = 'gabp')
    args = parser.parse_args()
    main(args.mode)
