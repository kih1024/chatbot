import tensorflow as tf
import numpy as np
import math
import sys

from Rnn_chatbot.config import FLAGS
from Rnn_chatbot.model import Seq2Seq
from Rnn_chatbot.dialog import Dialog
import xml.etree.ElementTree as ET
import urllib.request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

url = "http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.xml?key="
key = "2a83ee607d889ae32fca2cf9edbbe573"
url = url + key

class ChatBot:

    def __init__(self, voc_path, vector_path, train_dir): #!
        self.dialog = Dialog() # dialog 객체 생성
        self.dialog.load_vocab(voc_path, vector_path) # dataset에서 문장들을 한 줄씩 읽고 단어장을 초기화해준다. #! chat.voc과 word_embedding.voc을 확인.

        self.model = Seq2Seq(self.dialog.vocab_size) # 인코딩, 디코딩 RNN 신경망들을 Deep, Wide하게 만들어주고, 모델을 생성시킨다.
        # self.model = Seq2Seq(200)

        self.sess = tf.Session() # 세션.. Run 시켜줌.
        tf.reset_default_graph() # 초기 그래프 리셋
        ckpt = tf.train.get_checkpoint_state(train_dir) # 트레이닝 횟수 저장
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path) # Variable값을 불러와서 초기화해준다.
## 모델 만들고 세션을 실행하는데, 그래프 만들고 나서 다시 받아오기 위해 saver에 저장해둔다.

    def run(self, sentence): # 챗봇 구동 $$$$$
        # sys.stdout.write("> ")
        # sys.stdout.flush()
        # line = sys.stdin.readline()
        line = sentence  # $$$$$

        while line:
            print(self.get_replay(line.strip())) ###

            sys.stdout.write("\n> ")
            sys.stdout.flush()

            line = sys.stdin.readline()

    def decode(self, enc_input, dec_input):
        if type(dec_input) is np.ndarray:
            dec_input = dec_input.tolist() # 리스트로 변환
        # print("enc_input in decode : ", enc_input,"dec_input in decode : ",dec_input)
        # TODO: 구글처럼 시퀀스 사이즈에 따라 적당한 버킷을 사용하도록 만들어서 사용하도록
        if(len(enc_input) % 5 != 0):
            input_len = int(((len(enc_input)//5)+1)*5) # input의 길이를 설정 (5단위로 버켓팅해준다.)
        else:
            input_len = len(enc_input) # 인코딩 input의 길이가 5의 배수라면 길이 그대로 설정

        # dec_input_len = int(((len(dec_input) // 5) + 1) * 5) #decoding input의 길이를 설정 (5단위로 버켓팅해준다.)
        # print("input_len : ", input_len)

        enc_input, dec_input, _ = self.dialog.transform(enc_input, dec_input,
                                                        input_len,
                                                        FLAGS.max_decode_len) #패딩과 one-hot vector 생성

        return self.model.predict(self.sess, [enc_input], [dec_input]) #세션 실행

    def get_replay(self, msg): # msg : 내가 입력한 문장
        enc_input = self.dialog.tokenizer(msg, False) #문장에서 단어를 나눠준다.
        enc_input = self.dialog.tokens_to_ids(enc_input)  #토큰화된 단어에 리스트를 입력으로 넣어준다. 단어사전에 없는 단어는 Unknown처리
        dec_input = []

        # TODO: 구글처럼 Seq2Seq2 모델 안의 RNN 셀을 생성하는 부분에 넣을것
        #       입력값에 따라 디코더셀의 상태를 순차적으로 구성하도록 함
        #       여기서는 최종 출력값을 사용하여 점진적으로 시퀀스를 만드는 방식을 사용
        #       다만 상황에 따라서는 이런 방식이 더 유연할 수도 있을 듯
        curr_seq = 0
        for i in range(FLAGS.max_decode_len): #20개까지 output을 낼 수 있다.
            # print("enc_input : ", enc_input, " , dec_input : ", dec_input)
            outputs = self.decode(enc_input, dec_input)  #패딩 및 One-hot vector생성 후 세션 실행
            # print("outputs : ", outputs)
            if self.dialog.is_eos(outputs[0][curr_seq]): #결과값이 나온다면 break (target)
                break
            elif self.dialog.is_defined(outputs[0][curr_seq]) is not True: #Pre-defined에 정의되어 있지 않다면
                dec_input.append(outputs[0][curr_seq]) #인코딩 결과에 대해서 단어 하나를 디코딩 input값으로 넣어준다.
                curr_seq += 1

        reply = self.dialog.decode([dec_input], True)

        # if self.dialog.keyword :
        #     utf_keyword = str(self.dialog.keyword[0].encode('utf-8'))[2:-1].replace('\\x', '%')
        #     real_reply = url + "&movieNm=" + utf_keyword
        #
        #     tree = ET.ElementTree(file=urllib.request.urlopen(real_reply))
        #     root = tree.getroot()
        #
        #     reply += "\n총 " + str(len(root[1])) + "개의 영화가 있습니다.\n"
        #
        #     count = ""
        #     for i in range(0, len(root[1])):
        #         if i < len(root[1]) - 1:
        #             count = count + root[1][i][1].text + "\n"
        #         else:
        #             count = count + root[1][i][1].text + "\n"
        #
        #     reply += count
        #     self.dialog.keyword = []
        return reply


def main(_, sentence):  # $$$$$
    print("깨어나는 중 입니다. 잠시만 기다려주세요...\n")

    chatbot = ChatBot(FLAGS.voc_path, FLAGS.vec_path, FLAGS.train_dir) #! chat.voc, word_embedding.voc을 인자로 넣고, model폴더 안의 데이터들을 확인.
    chatbot.run(sentence)

if __name__ == "__main__":
    #tf.reset_default_graph()
    tf.app.run()
