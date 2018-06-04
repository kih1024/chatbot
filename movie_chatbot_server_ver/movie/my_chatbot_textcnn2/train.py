import os
import sys
import json
import time
import logging  # 커맨드 창에 로그 띄우는거.
import data_helper
import numpy as np
import tensorflow as tf
import mytoken
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from konlpy.tag import Mecab  # twitter
from gensim.models import Word2Vec  # word2vec
from gensim.models import FastText  # fasttext

logging.getLogger().setLevel(logging.INFO)

my_min = 0  # by odg


# 전처리 하는 과정.
def train_cnn():
    global my_min  # by odg

    # 오리지널 파일로부터 입력과 출력값을 배열로 만들어내기. 또한 CNN의 설정값 또한 파일로서 읽기
    """Step 0: load sentences, labels, and training parameters"""
    # 파라미터로 받은 파일을 로딩해서 문장배열(x_raw)과 각 문장들의 분류값배열(y_raw)을 얻어낸다. x는 뉴럴넷의 input이로, y는 output이다.
    train_file = sys.argv[1]
    # CNN네트워크의 각종 세부 설정값(hyper parameter)들을 로딩한다. 이 안에는 num_epochs, batch_size, num_filters등의 값들이 들어있다.
    x_raw, y_raw, df, labels = data_helper.load_data_and_labels(train_file)  # @
    # x_raw는  데이터셋, y_raw는 label의 One-hot vector, df는 라벨 포함 데이터셋, labels는 라벨 들어가있음.

    parameter_file = sys.argv[2]
    params = json.loads(open(parameter_file).read())

    model_dir = sys.argv[3]  # 모델 폴더 이름
    max_document_length = 0
    minimum_frequency = 5  # 단어장에 넣을 단어의 최소한의 빈도수(해당 빈도수 이상 있어야 단어장에 등록)
    list_max_final_scores = []  # final_scores 들 중 가장 큰 값만 저장한 리스트 by odg

    if (model_dir == "new"):
        timestamp = str(int(time.time()))
        model_name = "./trained_model_" + timestamp
        # 학습내용을 기록할 디렉토리 정의
        out_dir = os.path.abspath(os.path.join(os.path.curdir, model_name))  # !새롭게 생기는 폴더.
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        vectorize_list = list(mytoken.tokenizer(x_raw))

        for i in vectorize_list:
            if max_document_length < len(i):
                max_document_length = len(i)

        word2Vec = Word2Vec(vectorize_list, size=params['embedding_dim'] - params['num_of_class'], window=3,
                            min_count=minimum_frequency, workers=4)
        word2Vec.save(model_name + "/word2Vec.vec")  # @
        fastText = FastText(vectorize_list, size=params['embedding_dim'] - params['num_of_class'], window=3,
                            min_count=minimum_frequency, workers=4)
        fastText.save(model_name + "/fastText.vec")  # @

        vocab_dict, _ = data_helper.build_vocab(max_document_length, word2Vec.wv.index2word, params['num_of_class'],
                                                True)
        # 학습내용중 Tensorflow 내부의 변수상태가 저장됨 (예:AdamOptimizer)

    else:
        out_dir = os.path.abspath(os.path.join(model_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        vocab_dict, max_document_length = data_helper.build_vocab(0, None, None, False)
        model_name = model_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")

    # 전체 문장을 동일한 크기로 맞추고, 단어마다 ID를 부여해서, ID로 이루어진 문장을 만들기
    """Step 1: pad each sentence to the same length and map each word to an id"""
    # 문장배열의 크기를 pad값을 사용해서 같은 크기로 맞추어 주고, 문장안의 단어들을 ID로 매핑시켜주는 작업을 통해 학습 문장을 숫자 매트릭스형태로 만들어 학습이 가능한 상태로 만든다.
    logging.info('가장 긴 길이의 문장: {}'.format(max_document_length))  # 21

    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length,
                                                              vocabulary=vocab_dict,
                                                              tokenizer_fn=mytoken.tokenizer)  # 데이터셋의 단어들에 대해 인덱스를 붙여주는... #!

    x = np.array(list(vocab_processor.transform(x_raw)))  # !
    vocab_dictionary = vocab_processor.vocabulary_._mapping  # !
    y = np.array(y_raw)  # y는 라벨에 대한 One-hot vector

    # 데이터셋을 학습용과 테스트용으로 나누기
    """Step 2: split the original dataset into train and test sets"""
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    # 학습 문장과 결과값을 학습과 테스트 두개의 그룹으로 나눈다.(10%만 검증용으로 사용한다)
    # 데이터셋을 임의로 배치하고, 학습 데이터를 학습용과 검증용으로 다시 분류하기.
    """Step 3: shuffle the train set and split the train set into train and dev sets"""
    # 학습용 문장 배열(x_)의 순서를 그때마다 다르게 하기 위해 random방식으로 배열의 순서를 바꾸는 과정이다.
    # 학습데이터를 다시 두개의 그룹으로 나누는 것은 학습과 검증을 나눔으로서 overfitting을 줄이고 학습의 효과를 확인하기 쉽게 하기 위해서이다. 전체 데이터셋 구성은 다음과 같다. https://blog.naver.com/2feelus/221005831312 에서 확인.
    shuffle_indices = np.random.permutation(np.arange(len(y_)))  # 인덱스를 섞어줌
    x_shuffled = x_[shuffle_indices]  # 데이터셋에서 랜덤으로 셔플된 문장 인덱스 가져옴 ex) [5 69 0 ... 0]
    y_shuffled = y_[shuffle_indices]  # 해당 데이터셋에 대한 라벨 One-hot vector를 가져옴
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

    # 카테고리 라벨을 파일로 저장하여, 예측시에 활용할수 있도록 하기
    # 전체 카테고리 라벨들이 label.json 파일의 내용으로 저장. 실제 예측시에 이파일에 저장된 카테고리 순서에 따라 예측값을 얻어낸다.

    with open('./labels.json', 'w', encoding='utf-8-sig') as outfile:  # by odg
        json.dump(labels, outfile, indent=4, ensure_ascii=False)  # by odg

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    # 텐서 플로우 그래프생성이후 CNN 객체를 생성하기
    """Step 5: build a graph and cnn object"""
    # 텐서플로우에서 머신러닝의 데이터 흐름을 표현하는 그래프를 새로 생성한다. 그래프는 여러가지 머신러닝용 계산 명령 객체들을 포함하고 있다.
    graph = tf.Graph()
    # 파이선의 Context Manager 개념을 사용하여 기존의 기본 그래프 객체를 위에서 선언한 graph 객체로 대체하여 내부 블럭에 적용한다.
    # 멀티프로세스로 돌아가는 환경에서 이러한 방식을 사용하여 쓰레드에서 각각의 그래프 객체를 사용하도록 한다.
    with graph.as_default():
        # 세션을 새로 생성한다. 세션의 설정옵션으로 GPU를 특정하지 않기(allow_soft_placement=True),
        # 연산이 어느디바이스로 설정되었는지 보여주여주지 않기(log_device_placement=False)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.InteractiveSession(config=session_conf)  # !!
        # 세션또한 Context manager를 사용하여 세션의 열고 닫는 처리를 자동으로 해준다.
        with sess.as_default():
            # CNN객체를 생성한다. 파라미터 = 문장의 최대길이(sequence_length):912, 분류 카테고리수(num_classes):11,
            # 사전에 등록된 단어수(vocab_size):52943, 워드임베딩 사이즈(embedding_size):50,
            # CNN필터(커널)의 크기는 3x3,4x4,5x5 , 필터의 갯수는 총 32 개,
            # 오버피팅 방지를 위한 가중치 영향력 감소 수치(l2_reg_lambda):0.0
            cnn = TextCNN(
                sequence_length=x_train.shape[1],  # 들어온 문장의 최대 길이
                num_classes=y_train.shape[1],  # 라벨의 개수 (= One-hot vector의 길이)
                vocab_size=len(vocab_processor.vocabulary_),  # 단어의 수
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'],
                l2_reg_lambda=params['l2_reg_lambda'],
                vec_dir=model_name  # @
            )
            global_step = tf.Variable(0, name="global_step", trainable=False)  # !!원본 바꾼것
            # Cost function으로 Adam Optimizer사용
            optimizer = tf.train.AdamOptimizer(1e-3)
            # cnn의 loss(오차) 값을 파리미터로 받아 점진하강.
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # 학습에 사용할 함수 정의(session.run에서 사용됨).
            tf.summary.scalar("cnn_loss", cnn.loss)  # @@@
            tf.summary.scalar("cnn_accuracy", cnn.accuracy)  # @@@

            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # saver를 사용해 학습 내용을 저장
            saver = tf.train.Saver()

            # One training step: train the model with one batch
            # train_step 은 모델을 학습하는 하나의 묶음(batch)이다. 만약 batch size가 50이라면 50번의 Traning과 그에 따른 50번의 Test가 실행되게 된다.
            def train_step(x_batch, y_batch):
                # 입력/예측 출력값을 넣어줌으로서 학습/평가를 할수 있도록 한다.
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params[
                        'dropout_keep_prob']}  # Overfitting을 줄이기 위해,  Dropout(신경망 노드 탈락시키기) 확률을 지정.
                # 위에서 설정한 값들을 사용해 학습을 시작한다.
                _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

            # One evaluation step: evaluate the model with one batch
            # dev_step 은 학습 결과 묶음(batch)를 평가(Evaluation)하는 메소드이다.
            def dev_step(x_batch, y_batch):
                # 평가시에는 dropout은 사용하지 않는다.(dropout_keep_prob:1.0 => off)
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                # 평가시에는 학습용 train_op 파라미터는 넣지 않는다.
                step, loss, acc, num_correct, summary, scores, final_scores = sess.run(
                    [global_step, cnn.loss, cnn.accuracy, cnn.num_correct, merged, cnn.scores, cnn.final_scores],
                    feed_dict)  # @@@

                for j in final_scores:  # 각 final_scores에서 최대값 얻어오기
                    max_final_scores = max(j)
                    list_max_final_scores.append(max_final_scores)  # 각 최대값을 리스트에 추가

                min_final_scores = min(list_max_final_scores)  # 리스트에서 가장 작은 값
                writer.add_summary(summary, step)  # @@@

                return num_correct, min_final_scores

            # 사용된 단어들을 ID에 매핑시켜 차후 예측시에 사용한다.(학습시에는 사용하지 않음)
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            # 텐서플로우에서 사용하는 변수들을 초기화

            ckpt = tf.train.get_checkpoint_state(model_dir + "/checkpoints")  # checkpoint 얻는다.(모델의 Variable값을 얻어옴) #!!
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):  # 모델 checkpoint가 존재하면 #!!
                print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)  # !!
                saver.restore(sess, ckpt.model_checkpoint_path)  # checkpoint파일에서 모델의 변수값을 얻어온다. #!!
            else:  # 모델 checkpoint가 존재하지 않는다면 #!!
                print("새로운 모델을 생성하는 중 입니다.")  # !!
                sess.run(tf.global_variables_initializer())  # !!

            # Training starts here
            # 학습의 총 배치 갯수를 세팅한다. batch_iter 함수는 generator형식으로 작성되어있어서, 아래처럼 초기화를 해놓으면, for문안에서 배치단위로 값을 돌려주게 되어있다.
            # 한번에 학습단위묶음은 37개(batch_size=37). 학습데이터는 전체 학습에 한번 씩만 사용할것이다. (num_epochs=1)
            train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            # 최고의 정확성을 저장하기 위한 변수
            best_accuracy, best_at_step = 0, 0

            merged = tf.summary.merge_all()  # @@@
            writer = tf.summary.FileWriter("./logs", graph=graph)  # @@@

            """Step 6: train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                # zip을 사용하여 배치렬로 x(입력)과 y(기대출력)값을 각각 뽑아낸다.
                x_train_batch, y_train_batch = zip(
                    *train_batch)  # *는 unpack 하는거. https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean 참고.
                # 배치단위로 학습 진행
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                # 현재 학습 회차가 evaluate 할 순서이면 evaluate를 한x_dev다. 기본은 200번 마다.
                """Step 6.1: evaluate the model with x_dev and y_dev (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    # 개발용 데이터를 배치단위로 가져온다.
                    dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        # 학습된 모델에 개발용 배치 데이터를 넣어서 예측 성공 갯수를 누적한다.
                        num_dev_correct, _ = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct
                    # 모델의 정확성을 화면에 출력한다.
                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    # 가장 예측 확률이 좋게 나온 모델을 저장한다. 기준은 dev_accuracy가 가장 좋게 나온 step의 모델이다.
                    """Step 6.2: save the model if it is the best based on accuracy on dev set"""
                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)  # !!
                        tf.Print(path, [path], "This is saver : ")
                        logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy is {} at step {}'.format(best_accuracy, best_at_step))

            # 학습데이터와 Test데이터는 9:1로 나누었다.
            # Test데이터는 학습에 사용되지 않은 데이터로서, 학습된 모델이 객관성을 가지는지 확인하기 위한 용도이다
            """Step 7: predict x_test (batch by batch)"""
            test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct, min_final_scores = dev_step(x_test_batch, y_test_batch)
                my_min = min_final_scores  # by odg
                total_test_correct += num_test_correct

            f = open(checkpoint_dir + "_min.txt", "w")
            f.write(str(my_min))
            f.close()

            test_accuracy = float(total_test_correct) / len(y_test)

            logging.critical('테스트셋 Accuracy {}, best 모델 {}'.format(test_accuracy, path))  # 자꾸 여기서 오류남. 이건 그냥 log 띄우는거라 없어도 될거같은데..
            logging.critical('트레이닝 완료')


# def tokenizer(iterator):
#     tw = Twitter()
#     return (tw.morphs(x.strip()) for x in iterator)


# by odg
def return_my_min():
    return my_min


if __name__ == '__main__':
    train_cnn()
