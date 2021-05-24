import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# 加载数据集
# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

"""
求出预测值
"""
def predict(x,classifiers,alphas):
    ypredict = np.zeros([len(x)]).astype("float")
    for classifier, alpha in zip(classifiers, alphas):
        ypredict += alpha * classifier.predict(x)
    ypredict = np.sign(ypredict)
    return ypredict

# adaboost实现
"""
1.初始化训练数据的权值分布，n个训练样本数据，那么每一个训练样本最开始时都赋予相同的权值，w1=1/n
2.训练弱分类器hi。如果某个训练样本点，被弱分类器hi准确地分类，那么在构造下一个训练集中，它对应的权值要减小；
  如果某个训练样本点被错误分类，那么它的权值就应该增大。
  权值更新过的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去
3.将各个训练得到的弱分类器组合成一个强分类器，加大分类误差率小的弱分类器的权重 
  保证误差率低的弱分类器在最终分类器中占的权重较大，否则较小
"""

data , kind =create_data()  # data存放数据，kind 存放数据分类属性（-1,1）
lr = 0.6  # 学习率为0.6
classifiers = []  # 子分类器集合
alphas = []       # 子分类器权值
length_data = len(data)   # 样本个数
weights = np.array([1 / length_data] * length_data)  # 数据权重
classifier=DecisionTreeClassifier(max_depth=1)  # 调用函数DecisionTreeClassifier,决策树深度设为1
""""
训练弱分类器同时更新权重
"""
for i in range(50):  # 循环50次
    classifier.fit( data, kind ,sample_weight=weights)  # 训练子分类器
    y_new_predict = classifier.predict(data)  # 子分类器预测

    error_rate = np.sum( (y_new_predict != kind) * weights) / np.sum(weights)  # 计算加权错误率
    alpha = 0.5 * lr * np.log( (1 - error_rate) / error_rate )  # 计算alpha

    weights *= np.exp(-alpha * y_new_predict * kind)
    weights /= np.sum(weights)  # 更新数据权重

    classifiers.append(classifier)   # 收集子分类器
    alphas.append(alpha)             # 收集alpha

print(classifiers)
""""
画图
"""
y_predict=predict(data,classifiers,alphas)
error_rate = np.sum(y_predict != kind)/length_data # 算精度
fig = plt.figure(1)

xmin, xmax = np.min(data[:,0]-0.5, axis=0), np.max(data[:,0]+0.5, axis=0) # 算x,y轴界限
ymin, ymax = np.min(data[:,1]-0.5, axis=0), np.max(data[:,1]+0.5, axis=0)

test_X,test_Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j ] #生成网络采样点
grid_test = np.stack((test_X.flat,test_Y.flat) ,axis=1)   #测试点
# print(grid_test)
grid_hat = predict(grid_test,classifiers,alphas)          # 预测分类值
# print(grid_hat)
grid_hat = grid_hat.reshape(test_X.shape)                 # 使之与输入的形状相同

ax = fig.add_subplot(1, 1, 1)

# 为了可以成功显示汉字
matplotlib.pyplot.rcParams['font.sans-serif']=['SimHei']
matplotlib.pyplot.rcParams['axes.unicode_minus'] = False

ax.set( title='鸢尾花两个特征实现Adaboost分类(iter_num:{},error_rate:{})'.format( len(alphas), error_rate ))
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

cm_light=matplotlib.colors.ListedColormap(['#DA70D6', '#8B0000', '#00CED1']) # 配置颜色
ax.pcolormesh(test_X, test_Y, grid_hat, cmap=cm_light)                       # 预测值的显示
ax.scatter(data[kind==-1][:, 0], data[kind==-1][:, 1], marker='o')
ax.scatter(data[kind==1][:, 0], data[kind==1][:, 1], marker='x')     # 训练点的散点图
plt.show()

