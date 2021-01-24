from keras.models import load_model
import numpy as np
import tensorflow as tf

# 加载训练好模型
model = load_model('rnn_train_model.h5')

# 加载数据
test_texts, test_labels = 0, 0
filename = ['test_texts', 'test_labels']
for fil in filename:
    globals()[fil] = np.load(fil + '.npy')

# 乱序
np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
np.random.shuffle(test_texts)
np.random.seed(116)
np.random.shuffle(test_labels)
tf.random.set_seed(116)

# 将数据代入模型测试
test_loss, test_acc = model.evaluate(test_texts, test_labels, verbose=1)
print('Test accuracy:', test_acc)