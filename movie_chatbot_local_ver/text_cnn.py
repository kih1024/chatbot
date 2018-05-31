import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.models import FastText


class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, vec_dir=None):#, batch_size=0):#, voc_dic=None): #@
        """
            sequence_length는 문장안의 단어 수
            num_classes는 나눌 분류 개수
            vocab_size는 word2vec을 위한 단어장크기
            embedding_size
            filter_sizes는 convolution 필터 크기
            num_filters는 convolution 채널 수
            word2vec_dir은 word2vec시킨 벡터값들의 주소
        """
        with tf.device('/gpu:0'):
            # Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
            self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
            self.word2vec = Word2Vec.load(vec_dir + "/word2Vec.vec")  # by taeuk
            self.fasttext = FastText.load(vec_dir + "/fastText.vec")  # by taeuk
            self.label_one_hot_vec = np.eye(M=num_classes, N=num_classes + 1, k=0,
                                        dtype=np.float32)  # 각 라벨에 대한 One-hot vector 해줌 by taeuk
            self.label_vec = []  # by taeuk
            with open("./data/keyword_vec.vec", 'r', encoding='utf-8-sig') as vec_file:  # 단어장을 연다. by taeuk
                for line in vec_file:  # by taeuk
                    self.label_vec.append(self.label_one_hot_vec[int(line.strip())])  # by taeuk
            self.label_vec = np.array(self.label_vec)  # by taeuk
            self.scores = [[]]
            self.final_scores = 0.0
            models = ["Random", "Word2Vec", "FastText"]  # by taeuk

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

            for i, model in enumerate(models):
                # Embedding layer
                with tf.device('/cpu:0'), tf.name_scope('embedding-%s' % model):  # 작업에 사용할 cpu 지정. /cpu:0은 컴퓨터의 CPU를 의미
                    if(model == "Random"):
                        random = np.random.uniform(size=[vocab_size - 1, embedding_size - num_classes], low=-1.0, high=1.0)  # by taeuk
                        W = tf.Variable(np.vstack([[np.zeros(embedding_size, dtype=np.float32)],
                                                   np.concatenate((random, self.label_vec), axis=1)]), dtype=np.float32,
                                        name='W-%s' % model)
                    elif(model == "Word2Vec"):
                        W = tf.Variable(np.vstack([[np.zeros(embedding_size, dtype=np.float32)],
                                                   np.concatenate((self.word2vec.wv.vectors, self.label_vec), axis=1)]),
                                        dtype=np.float32, name='W-%s' % model)
                    elif(model == "FastText"):
                        W = tf.Variable(np.vstack([[np.zeros(embedding_size, dtype=np.float32)],
                                                   np.concatenate((self.word2vec.wv.vectors, self.label_vec), axis=1)]),
                                        dtype=np.float32, name='W-%s' % model)

                    self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
                    self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

                # 각 filter 크기 마다 convolution + maxpool 계층 생성
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):  # 3, 4, 5 순으로 필터 사이즈
                    # filter_size는 컨볼루션을 적용할 단어 수
                    # K(ksize)는 문장안에서 피쳐를 뽑아낼 단어집합
                    with tf.name_scope('conv-maxpool-%s-%s' % (filter_size, model)):
                        # Convolution Layer
                        filter_shape = [filter_size, embedding_size, 1, num_filters]  #
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W-%s' % model)
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b-%s' % model)
                        conv = tf.nn.conv2d(
                            self.embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='conv-%s' % model)

                        # non-linearity 적용
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu-%s' % model)

                        # outputs maxpooling
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, sequence_length - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='pool-%s' % model)
                        pooled_outputs.append(pooled)

                # 모든 pool features 결합
                num_filters_total = num_filters * len(filter_sizes)
                self.h_pool = tf.concat(pooled_outputs, 3)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

                # dropout 더함
                with tf.name_scope('dropout-%s' % model):
                    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

                # 최종 scores, predictions
                with tf.name_scope('output-%s' % model):
                    W = tf.get_variable(
                        'W-%s' % model,
                        shape=[num_filters_total, num_classes],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b-%s' % model)
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    if(model == 'Random'):
                        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores-%s' % model)
                    else:
                        self.scores = tf.add(self.scores, tf.nn.xw_plus_b(self.h_drop, W, b, name='scores-%s' % model))

            self.final_scores = tf.add(self.scores, 0.0, name='final_scores')  # 최종 self.scores 값 알아보려고 추가한거 by odg
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

            # Calculate mean cross-entropy loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y, logits = self.scores)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

            with tf.name_scope('num_correct'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='num_correct')
