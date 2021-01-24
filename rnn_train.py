import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 导入数据
train_texts,train_labels = 0,0
filename = ['train_texts', 'train_labels']
for fil in filename:
    globals()[fil] = np.load(fil + '.npy')
# 乱序
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(train_texts)
np.random.seed(116)
np.random.shuffle(train_labels)
tf.random.set_seed(116)
# 构建模型
model = keras.models.Sequential()
# output_dim:一个单词转为32维,input_dim:出现最频繁的4000个单词,input_length:一句评论的长度为400(不够的补0,多的去掉)
model.add(keras.layers.Embedding(output_dim=32, input_dim=4000, input_length=400))
# 用rnn不需要平坦层,加rnn
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=4)))
# 全连接层,256个神经元
model.add(keras.layers.Dense(units=32, activation='relu'))
# 丢弃50%神经元
model.add(keras.layers.Dropout(0.5))
# 输出层,使用softmax激活函数
model.add(keras.layers.Dense(units=2, activation='softmax'))
model.summary()

# 模型设置与训练
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# validation_split:划分率0.2,20%用来做验证;epochs:训练14个周期;batch_size:一次性训练样本数为128;verbose:训练过程可视化参数
history = model.fit(train_texts, train_labels, validation_split=0.2, epochs=14, batch_size=128, verbose=1)

# 训练过程可视化
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 保存模型
model.save('rnn_train_model.h5')
del model
