# 자세한 설명은 상위 폴더의 03 - Seq2Seq.py 등에서 찾으실 수 있습니다.
import tensorflow as tf
import tensorflow.contrib.layers as ly #!
from Rnn_chatbot.config import FLAGS #!
import numpy as np #!

# Seq2Seq 기본 클래스
class Seq2Seq:

    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self, vocab_size, n_hidden=128, n_layers=3):
        self.learning_late = 0.001

        self.vocab_size = vocab_size # 단어장 사이즈 변수
        self.n_hidden = n_hidden # hidden layer 계층 수 변수
        self.n_layers = n_layers # vertical하게 쌓인 RNN 계층의 depth

        #self.enc_input = tf.placeholder(tf.float32, [None, None, self.vocab_size]) #인코딩 input
        #self.dec_input = tf.placeholder(tf.float32, [None, None, self.vocab_size]) #디코딩 input
        self.enc_input = tf.placeholder(tf.float32, [None, None, FLAGS.embedding_size])  # 인코딩 input #!
        self.dec_input = tf.placeholder(tf.float32, [None, None, FLAGS.embedding_size]) #디코딩 input #!
        self.targets = tf.placeholder(tf.int64, [None, None]) # taget값

        # self.weights = tf.Variable(tf.ones([self.n_hidden, self.vocab_size]), name="weights")
        self.weights = tf.get_variable("weights", shape = [self.n_hidden, self.vocab_size], initializer=ly.xavier_initializer()) #! Xavier initializer로 초기화한 weight
        # self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.bias = tf.Variable(tf.random_normal([self.vocab_size]), name="bias") #! random하게 초기화한 bias
        self.global_step = tf.Variable(0, trainable=False, name="global_step") #전체 횟수

        self.build_model() #모델을 만들어준다.

        self.saver = tf.train.Saver(tf.global_variables()) # 변수 저장

    def build_model(self):
        #self.enc_input = tf.transpose(self.enc_input, [1, 0, 2])
        #self.dec_input = tf.transpose(self.dec_input, [1, 0, 2])

        enc_cell, dec_cell = self.build_cells() #인코더와 디코더에 대한 cell들을 build해준다.

        with tf.variable_scope('encode'):
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, self.enc_input, dtype=tf.float32) #encoding RNN신경망을 만들어준다.

        with tf.variable_scope('decode'):
            outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, self.dec_input, dtype=tf.float32,
                                                    initial_state=enc_states) #인코딩 상태를 초기 상태로 초기화 해주고 decoding RNN신경망을 만들어준다.

        self.logits, self.cost, self.train_op = self.build_ops(outputs, self.targets) # 모델 교육에 있어 필요한 수식, cost, 최적화된 값..

        self.outputs = tf.argmax(self.logits, 2) # logit에서 2차원의 요소 중에서 큰 값을 골라서 해당 인덱스를 output에 넣어준다.

    def cell(self, n_hidden, output_keep_prob):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(self.n_hidden) # 베이직한 RNN 셀 생성
        # rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)  # 베이직한 LSTM 셀 생성 #!
        # rnn_cell = tf.contrib.rnn.GRUCell(self.n_hidden) # 베이직한 GRU 셀 생성 #!
        # 원래 BasicRNNCell보다는 최근에 Rnn에서 파생된 LSTMCell이나 GRUCell을 많이 쓰지만 일단 트레이닝과정에서 가장 좋은 성능을 보였던 RNNCell로 사용하고 있습니다.
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob) #Dropout해준다(Overfitting 방지하기 위해)
        return rnn_cell #Rnn cell 리턴

    def build_cells(self, output_keep_prob=0.7):
        enc_cell = tf.contrib.rnn.MultiRNNCell([self.cell(self.n_hidden, output_keep_prob)
                                                for _ in range(self.n_layers)])
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.cell(self.n_hidden, output_keep_prob)
                                                for _ in range(self.n_layers)])
        # 인코더와 디코더의 셀들을 가로로 n_hidden만큼, 세로로 n_layers만큼 만든다.
        return enc_cell, dec_cell

    def build_ops(self, outputs, targets):
        time_steps = tf.shape(outputs)[1] #output의 1차원의 요소 개수
        outputs = tf.reshape(outputs, [-1, self.n_hidden]) # 리스트의 요소를 n_hidden개 만큼 가지도록 설정

        logits = tf.matmul(outputs, self.weights) + self.bias # logits = W * Outputs + B
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size]) # logit텐서에 대해서 3차원이고 제일 안쪽 요소를 vocab_size, time_step개 만큼 2차원..

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)) # softmax를 쓰고 평균 오차값의 cost를 구한다
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_late).minimize(cost, global_step=self.global_step) #Optimizer중에 좋은 성능을 보이는 AdamOptimizer로 최적화해준다.
        #cost를 줄이기 위해 최적화를 해준다.

        tf.summary.scalar('cost', cost) #cost값을 얻는다

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})
        # hypothesis = weights * outputs + bias로 지정된 hypothesis와 target값과의 비교를 통해 cost를 최저화해주는 모델에 enc_input, dec_input, target을 넣어서 세션을 돌린다.

    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.targets) # output값(즉, 챗봇 모델에서 출력된 예상답변)과 실제답변(target값)을 비교해서 유추한 답이 맞는지 확인
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32)) #위에서 체크한 답을 통해서 정확성을 측정

        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})
        # hypothesis = weights * outputs + bias로 지정된 hypothesis와 target값과의 비교를 통해 cost를 최저화해주는 모델에 enc_input, dec_input, target을 넣어서 세션을 돌려서 정확성을 얻어낼수 있다.

    def predict(self, session, enc_input, dec_input): #enc_input과 dec_input을 넣어서 세션 실행
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets): #log를 써준다.
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.targets: targets}) #모델에 대해서 세션을 돌려서 merge된 데이터에 대해서 얻을 수 있다.

        writer.add_summary(summary, self.global_step.eval()) #로그 써준다.
