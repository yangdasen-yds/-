"""
断点续训,对于未训练完的模型可以继续训练
"""

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 加载模型
model = load_model('rnn_train_model.h5')

# 导入数据
train_texts, train_labels = 0, 0
filename = ['train_texts', 'train_labels']
for fil in filename:
    globals()[fil] = np.load(fil + '.npy')

# 乱序
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(train_texts)
np.random.seed(116)
np.random.shuffle(train_labels)
tf.random.set_seed(116)

history = model.fit(train_texts, train_labels, validation_split=0.2, epochs=4, batch_size=32, verbose=1)

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

model.save('rnn_train_model.h5')
del model
