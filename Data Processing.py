import pandas as pd
import numpy as np

# 读取 CSV 文件
df = pd.read_csv('data_12k_1797_10c.csv')

# 定义信号间隔长度和每块样本点数
interval_length = 1024
samples_per_block = 1024


# 数据前处理函数
def Sampling(signal, interval_length, samples_per_block):
    num_samples = len(signal)
    num_blocks = num_samples // samples_per_block
    samples = []
    for i in range(num_blocks):
        start = i * samples_per_block
        end = start + interval_length
        samples.append(signal[start:end])
    return np.array(samples)


def DataPreparation(df, interval_length, samples_per_block):
    X, LabelPositional, Label = None, None, None
    for count, column in enumerate(df.columns):
        SplitData = Sampling(df[column].values, interval_length, samples_per_block)
        y = np.zeros([len(SplitData), 10])
        y[:, count] = 1
        y1 = np.zeros([len(SplitData), 1])
        y1[:, 0] = count
        # 堆叠并标记数据
        if X is None:
            X = SplitData
            LabelPositional = y
            Label = y1
        else:
            X = np.append(X, SplitData, axis=0)
            LabelPositional = np.append(LabelPositional, y, axis=0)
            Label = np.append(Label, y1, axis=0)
    return X, LabelPositional, Label


# 数据前处理
X, Y_CNN, Y = DataPreparation(df, interval_length, samples_per_block)

print('Shape of Input Data =', X.shape)
print('Shape of Label Y_CNN =', Y_CNN.shape)
print('Shape of Label Y =', Y.shape)



# k折交叉验证
from sklearn.model_selection import train_test_split, KFold

kSplits = 5
kfold = KFold(n_splits=kSplits, random_state=32, shuffle=True)

# 大家可以分别输出查看一下 X, Y_CNN, Y 长什么样，下面我展示一下
# X太长了就不放了；
# Reshape数据
Input_1D = X.reshape([-1, 1024, 1])

# 数据集划分
X_1D_train, X_1D_test, y_1D_train, y_1D_test = train_test_split(Input_1D, Y_CNN, train_size=0.75, test_size=0.25,
                                                                random_state=101)


# 定义1D-CNN模型
class CNN_1D():
    def __init__(self):
        self.model = self.CreateModel()

    def CreateModel(self):
        model = models.Sequential([
            layers.Conv1D(filters=16, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=64, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu'),
            layers.MaxPool1D(pool_size=2),
            layers.Flatten(),
            layers.InputLayer(),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(10),
            layers.Softmax()
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model


accuracy_1D = []

# 训练结果
for train, test in kfold.split(X_1D_train, y_1D_train):
    Classification_1D = CNN_1D()
    history = Classification_1D.model.fit(X_1D_train[train], y_1D_train[train], verbose=1, epochs=12)
    kf_loss, kf_accuracy = Classification_1D.model.evaluate(X_1D_train[test], y_1D_train[test])
    accuracy_1D.append(kf_accuracy)

CNN_1D_train_accuracy = np.average(accuracy_1D) * 100
print('CNN 1D train accuracy =', CNN_1D_train_accuracy)

CNN_1D_test_loss, CNN_1D_test_accuracy = Classification_1D.model.evaluate(X_1D_test, y_1D_test)
CNN_1D_test_accuracy *= 100
print('CNN 1D test accuracy =', CNN_1D_test_accuracy)