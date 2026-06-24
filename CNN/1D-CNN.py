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