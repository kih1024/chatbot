import os
import sys
import json
import logging
import data_helper
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from konlpy.tag import Mecab  # hashtag 얻어올라고
import o_actor_search  # 배우 검색용 - KMDB
import o_movie_name_search  # 영화 제목 검색용 - KMDB
import o_movie_release_date_search  # 영화 개봉년도 검색용 - KMDB
import o_movie_runtime_search  # 영화 상영시간 검색용 - KMDB
import o_movie_rating_search  # 영화 평점 검색 - Naver API
import o_movie_attendance_search
from Rnn_chatbot.chat import main  # Rnn_chatbot의 chat.py의 main 함수 호출
import mytoken
logging.getLogger().setLevel(logging.INFO)

# ^vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path) 이 부분에서 오류났었음.
# ^27줄의 tokenizer 함수는 태욱이형이 추가한 train.py 의 tokenizer 함수 그대로 가져온거.
# ^원래 33줄에 my_tokenizer 함수 이름이 tokenizer 였는데
# ^형이 보낸 train.py 39줄 vocab_processor 인자 중에 tokenizer 부분에 my_tokenizer 가 들어가면서 오류떴던거.
# ^정리하자면 33줄의 my_tokenizer 가 원래 tokenizer 였는데 앞에 my_ 만 붙여서
# ^vocab_processor 정상적으로 동작시킨거.

# def tokenizer(iterator):
#     tw = Twitter()
#     return (tw.morphs(x.strip()) for x in iterator)
#

# 해시태그(hash_tag, #) 뒤엣놈을 가져올 함수
def my_tokenizer(sentence):  # !
    tw = Mecab()  # 트위터 객체를 만들어준다.
    # 공백으로 나누고 특수문자는 따로 뽑아낸다.
    temp = sentence.split()
    keyword = []
    x = sentence
    for key in temp:
        if '#' in key:
            keyword.insert(0, key[1:].replace('_',"").replace('#',""))
            x = x.replace(key, '#')
    return keyword, x.strip()


def predict_unseen_data():
    """Step 0: load trained model and parameters"""
    params = json.loads(open('./parameters.json').read())
    checkpoint_dir = sys.argv[1]  # trained_model_숫자 폴더
    if not checkpoint_dir.endswith('/'):  # 이건 오타 방지용으로 / 없으면 / 붙여서 밑에 checkpoints 폴더에 접근하게!
        checkpoint_dir += '/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')  # checkpoints 폴더 안의 가장 최근 체크포인트 파일
    logging.critical('Loaded the trained model(checkpoint_file): {}'.format(checkpoint_file))

    """Step 1: load data for prediction"""
    sys.stdout.write("> ")
    sys.stdout.flush()
    test_file = sys.stdin.readline()  # 얘가 test_examples 의 역할 by odg

    # labels.json was saved during training, and it has to be loaded during prediction
    labels = json.loads(open('./labels.json', encoding='utf-8-sig').read())  # labels.json 읽어온거 전체 문자열.  # by odg
    one_hot = np.zeros((len(labels), len(labels)), int)  # labels 길이(3) * labels 길이(3) 만큼의 행렬을 만듬! 즉, 3x3 행렬. 각 요소는 0
    np.fill_diagonal(one_hot, 1)  # 이 행렬의 각 값은 1
    label_dict = dict(zip(labels, one_hot))  # type(label_dict) = <class 'dict'>
    x_raw = [0]
    x_raw[0] = test_file  # 데이터셋 전체 문자열이니까 입력값을 받아오는 test_file 과 동일해지겠네@
    my_keyword, x_raw[0] = my_tokenizer(x_raw[0])  # 해시태그(#, hashtag) 뒤에 키워드(keyword) 얻어온거@ list타입임@

    if not my_keyword :  # hashtag 없이 (키워드 없이) 쳤을 때  $$$$$
        logging.critical('키워드 없음')
        main("", x_raw[0])
        exit()

    str_my_keyword = str(my_keyword[0])  # str_my_keyword 는 검색할 때 str 타입으로 넣어줘야 되서 형변환만 한거@
    x_test = [x for x in x_raw]  # clean_str 안씀! 요놈때문에 예측률이 안좋게 나왔던거!!
    
    logging.info('x_test 개수 : {}'.format(len(x_test)))

    vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")  # vocab_path = ./trained_model_1522396321/vocab.pickle
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    x_test = np.array(list(vocab_processor.transform(x_test)))
    

    """Step 2: compute the predictions"""
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]  # input_x = Tensor("input_x:0", shape=(?, 7), dtype=int32)
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]  # dropout_keep_prob = Tensor("dropout_keep_prob:0", dtype=float32)
            predictions = graph.get_operation_by_name("predictions").outputs[0]  # predictions = Tensor("output/predictions:0", shape=(?,), dtype=int64) #@
            scores = graph.get_operation_by_name("final_scores").outputs[0]

            batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)  # batches = <generator object batch_iter at 0x000001F26679AD00
            all_predictions = []  # all_predictions 가 예측값
            for x_test_batch in batches:
                f = open(checkpoint_dir + "checkpoints_min.txt", "r")  # by odg
                str_my_min = f.readline()
                #float_my_min = float(str_my_min)  # float_my_min 이 train 한 것들 중 가장 작은 값 by odg
                float_my_min = float(4)  # float_my_min 이 train 한 것들 중 가장 작은 값 by odg
                print("float_my_min : ", float_my_min)

                sco = sess.run(scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})  # 수정$$
                for i in sco:  # by odg
                    max_final_scores = max(i)

                print("max_final_scores : ", max_final_scores)

                if float_my_min > max_final_scores:
                    print("제대로 예측 못한 것.")
                    exit()  # by odg 이 부분만 챗봇으로 보내면 될듯.

                logging.critical('sco: {}'.format(sco))  # 수정$$
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                logging.critical('batch_predictions: {}'.format(batch_predictions))
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                print("all_predictions : ", all_predictions)

    actual_labels = [labels[int(all_predictions)]]
    logging.info('예측값 actual_labels 입니다 : {}'.format(actual_labels))
    if actual_labels == ['영화 제목 검색']:  # 영화 제목 검색이면@
        o_movie_name_search.movie_name_search(str_my_keyword)
    if actual_labels == ['배우 검색']:  # 배우 검색이면@
        o_actor_search.actor_name_search(str_my_keyword)
    if actual_labels == ['영화 개봉년도 검색']:
        o_movie_release_date_search.movie_release_date_search(str_my_keyword)
    if actual_labels == ['영화 상영시간 검색']:
        o_movie_runtime_search.movie_runtime_search(str_my_keyword)
    if actual_labels == ['영화 평점 검색']:
        o_movie_rating_search.movie_rating_search(str_my_keyword)
    if actual_labels == ['영화 누적관객수 검색']:
        o_movie_attendance_search.movie_attendance_search(str_my_keyword)

    logging.critical('The prediction is complete')

if __name__ == '__main__':
    predict_unseen_data()
