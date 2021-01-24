"""
对单个电影评论预测其是正面评论还是负面评论
"""
from keras.models import load_model
import re
import os
import keras
import numpy as np


# 删除文本中<br />这类html标签
def remove_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# 和预处理时生成的token一样
def read_files(filetype):
    path = 'data/aclImdb/'
    file_list = []

    # 读取正面评价的文件的路径,存到file_liat列表里
    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    # 读取负面评价的文件的路径,存到file_liat列表里
    negative_path = path + filetype + '/neg/'
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    # 得到所有文本
    all_texts = []
    for fil in file_list:
        with open(fil, 'r', encoding='utf-8') as file_input:
            # 文本中有<br />这类html标签,将文本传入remove_tags函数
            # 函数里使用正则表达式可以将这样的标签清楚掉
            all_texts += [remove_tags(''.join(file_input.readline()))]
    return all_texts


def forecast(texts):
    sentiment_dict = {0: 'pos', 1: 'neg'}
    print('电影评论为:', texts)
    model = load_model('rnn_train_model.h5')
    texts = remove_tags(texts)
    input_seq = token.texts_to_sequences([texts])
    pad_input_seq = keras.preprocessing.sequence.pad_sequences(input_seq, padding='post', truncating='post', maxlen=400)
    pred = model.predict(pad_input_seq)
    print('预测结果为:', sentiment_dict[np.argmax(pred)])


# 得到 训练与测试用的标签和文本
train_texts = read_files('train')
# 建立词汇词典Token
token = keras.preprocessing.text.Tokenizer(num_words=4000)
token.fit_on_texts(train_texts)

# 需要预测的评论
texts = 'This is a very strange film that was long thought to be forgotten. Its the story of two American Army ' \
        'buddies, William Boyd (aka "Hopalong Cassidy") and Louis Wolheim, and their adventures as they manage to ' \
        'escape from a German prison camp during WWI. However, as this is a comedy, the duo manage to make the most ' \
        'round about and stupid escape--accidentally boarding a train to Constantinople to be placed in a Turkish ' \
        'prisoner of war camp! On the way, they manage to escape once again and end up in quite the ' \
        'adventure--meeting sexy Mary Astor along the way.<br /><br />As far as the film goes, it was a rather funny ' \
        'script and despite being a silly plot, it worked rather well. The chemistry between Boyd and Wolheim worked ' \
        'and the film managed to be quite entertaining. Oddly, however, the film managed to beat out Harold Lloyds ' \
        'film, SPEEDY, for an Oscar for Best Direction for a Comedy (a category no longer used)--as SPEEDY was a ' \
        'superior film in most ways (its one of Lloyds best films). Still, its well worth a look--especially if you ' \
        'love silent films.<br /><br />By the way, director Milestone and Louis Walheim would team up just a few ' \
        'years later for another WWI picture, the great ALL QUIET ON THE WESTERN FRONT--a film that is definitely NOT ' \
        'a comedy. '

forecast(texts)
