"""
下载数据aclImdb_v1.tar.gz到data文件夹,然后解压
"""

import os
import tarfile
import urllib.request


# 数据地址
url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'

filepath = 'data/aclImdb_v1.tar.gz'

# 下载数据
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.isfile(filepath):
    print('download...')
    result = urllib.request.urlretrieve(url, filepath)
    print('downloaded:', result)
else:
    print(filepath, 'is existed')

# 解压数据
if not os.path.exists('data/aclImdb'):
    tfile = tarfile.open(filepath, 'r:gz')
    print('extracting...')
    tfile.extractall('data/')
    print('extraction completed')
else:
    print('data/aclImdb is existed')
