# Vector Representations of Words
## 背景介绍

## 基础版Word2Vec
```
python word2vec_basic.py
```


## 提高版Word2Vec
如果已经运行过基础版Word2Vec，那么训练数据已经被下载下来了，否则可以通过下面命令下载数据：
```
wget http://mattmahoney.net/dc/text8.zip
```

解压准备好的训练数据：
```
unzip text8.zip
```

通过运行训练程序：
```
python word2vec.py --train_data=text8 --eval_data=questions-words.txt --save_path=/tmp --interactive=true
```

单机环境下，这个程序需要运行几个小时到10几个小时。


## Words Calculation

