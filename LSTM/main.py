import numpy as np
import pandas as pd
import sklearn
from joblib import dump, load

# 时间步长 1024 和 重叠率 -0.5 
# window = 1024  step = 512

def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):
    """
        data:要切分的时间序列数据,可以是一个一维数组或列表。
        time_steps:切分的时间步长,表示每个样本包含的连续时间步数。
        lable: 表示切分数据对应 类别标签
        overlap_ratio:前后帧切分时的重叠率,取值范围为 0 到 1,表示重叠的比例。
    """
    stride = int(time_steps * (1 - overlap_ratio))  # 计算步幅
    samples = (len(data) - time_steps) // stride + 1  # 计算样本数
    # 用于存储生成的数据
    Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])  
    # 记录数据行数(量)
    index_count = 0 
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(lable)  # 对应哪一类
        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
    return Clasiffy_dataFrame

# 归一化数据
def normalize(data):

    s= (data-min(data)) / (max(data)-min(data))
    return  s

# 数据集的制作
def make_datasets(data_file_csv, split_rate = [0.7,0.2,0.1]):

    # 1.读取数据
    origin_data = pd.read_csv(data_file_csv)
    # 2.分割样本点
    time_steps = 1024  # 时间步长
    overlap_ratio = 0.5  # 重叠率
    # 用于存储生成的数据# 10个样本集合
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])  
    # 记录类别标签
    label = 0
    # 使用iteritems()方法遍历每一列
    for column_name, column_data in origin_data.items():
        # 对数据集的每一维进行归一化
        # column_data = normalize(column_data)
        # 划分样本点  window = 512  overlap_ratio = 0.5  samples = 467 每个类有467个样本
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        label += 1 # 类别标签递增
        samples_data = pd.concat([samples_data, split_data])
        # 随机打乱样本点顺序 
        samples_data = sklearn.utils.shuffle(samples_data) # 设置随机种子 保证每次实验数据一致

    # 3.分割训练集、验证集、测试集
    sample_len = len(samples_data) # 每一类样本数量
    train_len = int(sample_len*split_rate[0])  # 向下取整
    val_len = int(sample_len*split_rate[1]) 
    train_set = samples_data.iloc[0:train_len,:]   
    val_set = samples_data.iloc[train_len:train_len+val_len,:]   
    test_set = samples_data.iloc[train_len+val_len:sample_len,:]   
    return  train_set, val_set, test_set, samples_data

# 生成数据集
train_set, val_set, test_set, samples_data = make_datasets('data_12k_10c.csv')

# 保存数据
dump(train_set, 'train_set') 
dump(val_set, 'val_set') 
dump(test_set, 'test_set') 