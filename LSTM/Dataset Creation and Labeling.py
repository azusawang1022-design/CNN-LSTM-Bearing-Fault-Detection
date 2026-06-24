import torch
from joblib import dump, load
def make_data_labels(dataframe):

    # 信号值
    x_data = dataframe.iloc[:,0:-1]
    # 标签值
    y_label = dataframe.iloc[:,-1]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64')) # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return x_data, y_label

# 加载数据
train_set = load('train_set')
val_set = load('val_set')
test_set = load('test_set')

# 制作标签
train_xdata, train_ylabel = make_data_labels(train_set)
val_xdata, val_ylabel = make_data_labels(val_set)
test_xdata, test_ylabel = make_data_labels(test_set)

# 保存数据
dump(train_xdata, 'trainX_1024_10c')
dump(val_xdata, 'valX_1024_10c')
dump(test_xdata, 'testX_1024_10c')
dump(train_ylabel, 'trainY_1024_10c')
dump(val_ylabel, 'valY_1024_10c')
dump(test_ylabel, 'testY_1024_10c')