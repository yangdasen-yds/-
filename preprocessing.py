"""
数据预处理,将处理后的数据保存为npy格式文件('train_texts(25000,400)',
'test_texts(25000,400)', 'train_labels(25000,2)', 'test_labels(25000,2)')
"""
import re
import os
import keras
import json
import numpy as np


# 删除文本中<br />这类html标签
def remove_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# 读取文件,参数filetype取值为train或者test
def read_files(filetype):
    path = 'data/aclImdb/'
    file_list = []

    # 读取正面评价的文件的路径,存到file_liat列表里
    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]
    pos_file_num = len(file_list)

    # 读取负面评价的文件的路径,存到file_liat列表里
    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]
    neg_file_num = len(file_list) - pos_file_num

    print('read', filetype, 'files', len(file_list))
    print(pos_file_num, 'pos files in', filetype, 'files')
    print(neg_file_num, 'neg_files in', filetype, 'files')

    # 得到所有标签,标签用one-hot编码表示,正面评价标签为[1,0],负面评价标签为[0,1]
    all_labels = ([[1, 0]] * pos_file_num + [[0, 1]] * neg_file_num)

    # 得到所有文本
    all_texts = []
    for fil in file_list:
        with open(fil, 'r', encoding='utf-8') as file_input:
            # 文本中有<br />这类html标签,将文本传入remove_tags函数
            # 函数里使用正则表达式可以将这样的标签清楚掉
            all_texts += [remove_tags(''.join(file_input.readline()))]
    return all_labels, all_texts


# 得到 训练与测试用的标签和文本
train_labels, train_texts = read_files('train')
test_labels, test_texts = read_files('test')

# 查看数据标签
print('训练数据,正面评价例子 文本:', train_texts[0])
print('训练数据,正面评价例子 标签:', train_labels[0])
print('训练数据,负面评价例子 文本:', train_texts[12500])
print('训练数据,负面评价例子 标签:', train_labels[12500])

print('测试数据,正面评价例子 文本:', test_texts[0])
print('测试数据,正面评价例子 标签:', test_labels[0])
print('测试数据,负面评价例子 文本:', test_texts[12500])
print('测试数据,负面评价例子 标签:', test_labels[12500])

# 建立词汇词典Token
token = keras.preprocessing.text.Tokenizer(num_words=4000)
token.fit_on_texts(train_texts)
print(type(token))
with open('token.json', 'w', encoding='utf-8') as f:
    json.dump(token, f)
# 查看token读取了多少文档
# token.document_count

# 查看Token中词汇出现的频次排名
# print(token.word_index)


# 文字转数字列表
train_sequences = token.texts_to_sequences(train_texts)
test_sequences = token.texts_to_sequences(test_texts)

print(train_texts[0])
print(train_sequences[0])

# 让转换后的数字列表长度相同(不够数量的后面补0,超长的截取后面的)
train_texts = keras.preprocessing.sequence.pad_sequences(train_sequences, padding='post', maxlen=400)
test_texts = keras.preprocessing.sequence.pad_sequences(test_sequences, padding='post', maxlen=400)

print(train_texts.shape)

# 将处理后的数据保存为npy格式文件
filename = ['train_texts', 'test_texts', 'train_labels', 'test_labels']
for fi in filename:
    np.save(fi + '.npy', globals()[fi])
