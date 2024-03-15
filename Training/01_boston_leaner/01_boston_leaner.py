from typing import List
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel
from sklearn.utils import Bunch


"""
    1. 数据集      data set
    2. 回归模型     mode
    3. 训练       fit
    4. 预测       predict
    5. 绘制       show
"""
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体


# 使用 fetch_openml 加载数据集
def get_data_set() -> Bunch:
    data_set = fetch_openml(name='Boston', version=1, as_frame=False)
    return data_set


# 创建线性回归模型
def get_liner_mode() -> LinearModel:
    liner = LinearRegression()
    return liner


# 使用 RM (平均房间数) 特征进行训练
def fit_liner_mode(liner_mode: LinearModel, data_set: Bunch) -> None:
    liner_mode.fit(data_set.data[:, 5:6], data_set.target)


# 进行预测
def pre_diction(liner_mode: LinearModel, data_set: Bunch) -> List[float]:
    predictions: List[float] = liner_mode.predict(data_set.data[:, 5:6])
    return predictions


# 绘制散点图和线性回归线
def draw_graph(data_set: Bunch, predictions: List[float]):
    plt.scatter(data_set.data[:, 5], data_set.target, label='实际值')
    plt.xlabel('房间数')
    plt.ylabel('房价')
    plt.legend()

    plt.plot(data_set.data[:, 5], predictions, color='red', label='预测值')
    plt.show()


if __name__ == '__main__':
    data = get_data_set()
    liner = get_liner_mode()
    fit_liner_mode(liner, data)
    pre_dict = pre_diction(liner, data)
    print(f"pre_dict:{pre_dict}")
    draw_graph(data, pre_dict)

