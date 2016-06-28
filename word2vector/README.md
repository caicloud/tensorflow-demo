# Vector Representations of Words
## 背景介绍
传统的自然语言处理一般使用Bag-of-words模型，把每个单词当成一个符号。比如"cat"用Id123表示，"kitty"用Id456表示。用这样的方式表达单词的一个最大坏处是它忽略了单词之间的语义关系。同时Bag-of-words模型也会导致特征矩阵过于稀疏的问题。用向量来表示一个单词（word to vector, embedding）就可以从一定程度上解决这些问题。具体的Word2Vec的背景，方法和应用在[这篇文章](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)中都有详述，这里我们就不再赘述。下面我们需要介绍如何将Word2Vec算法在Tensorflow上跑起来以及Word2Vec的一个小应用。

## 基础版Word2Vec
```
python word2vec_basic.py
```

## I/O速度提高版
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
python word2vec.py --train_data=text8 --eval_data=questions-words.txt --save_path=/tmp 
```

单机环境下，这个程序可能需要运行10几个小时。

## 训练速度提高版
如果没有准备数据，那么可以通过上述方法下载数据，数据准备好之后运行：
```
python word2vec_optimized.py --train_data=text8 --eval_data=questions-words.txt --save_path=/tmp
```
相比上面的模型，这个方法可以加速~15-20倍。


## 实现单词加减法
#### 使用上面训练出来的向量
上面几个程序都没有输出最后每个单词得到的向量，如果想要使用上述结果，需要输出每个单词对应的向量，格式如下：
```
单词1 向量1
单词2 向量2
...
单词n 向量n
```

其中单词存在```self._options.vocab_words```中，每个单词对应的embedding在```self._emb``` (word2vec.py)，```self._w_in``` (word2vec_optimized.py)中。

#### 使用已经训练好的向量
网上有很多已经训练好的Word2Vec模型，其中stanford nlp实验室的[GloVe](http://nlp.stanford.edu/projects/glove/)就提供了不少模型。可以通过下述命令直接下载：
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

#### 运行单词计算器

