import tensorflow as tf
import numpy as np
import re
import codecs
import tensorflow.contrib.training as bk
from collections import OrderedDict #!
from itertools import repeat #!
from konlpy.tag import Twitter #!
from konlpy.utils import pprint #!
import matplotlib #!
import matplotlib.pyplot as plt #!
import tensorflow.contrib.layers as ly #!

# from config import FLAGS
from Rnn_chatbot.config import FLAGS

class Dialog():

    _PAD_ = "_PAD_"  # 빈칸 채우는 심볼
    _STA_ = "_STA_"  # 디코드 입력 시퀀스의 시작 심볼
    _EOS_ = "_EOS_"  # 디코드 입출력 시퀀스의 종료 심볼
    _UNK_ = "_UNK_"  # 사전에 없는 단어를 나타내는 심볼 / 예외 처리

    _PAD_ID_ = 0
    _STA_ID_ = 1
    _EOS_ID_ = 2
    _UNK_ID_ = 3
    _PRE_DEFINED_ = [_PAD_ID_, _STA_ID_, _EOS_ID_, _UNK_ID_]
    word_embed_dic = []

    def __init__(self):
        self.vocab_list = [] # 단어 리스트
        self.vocab_dict = {} # 단어 사전
        self.vocab_size = 0 # 사이즈 0으로 초기화
        self.each_vocab_size = [] #@ 한 단어 사전에 여러 대답과 질문이 있을 때 사용
        self.examples = [] # 단어 전체

        self.each_examples = []
        self.each_examples_size = []
        self.each_answer = []
        self.stopwords = []

        self.numberOfAnswer = 0 # 대답의 개수

        self.skip_grams = [] #!
        self.vocab_vec = [] #!
        self.sentence_vec = [] #!
        self.keyword = [] #!

        self._index_in_epoch = 1 #!

        openStopwords = open(FLAGS.stopwords, 'r')
        for line in openStopwords:
            line = line.strip()
            self.stopwords.append(line)
        openStopwords.close()
        print("stopwords : ", self.stopwords)

    def decode(self, indices, string=False):
        tokens = [[self.vocab_list[i] for i in dec] for dec in indices] # decode_input으로 받은 index들을 단어장에서 찾아서 토큰화해준다.
        # print("tokens : ", tokens)
        for i in range(0, len(tokens)): #!
            if '#' in tokens[i]: #! 토큰화된 단어에 #이 있다면(즉, 해시태그된 키워드가 있다면)
                # print("분기 들어옴")
                if self.keyword:
                    tokens[i][tokens[i].index('#')] = self.keyword[0] #! keyword list에 키워드를 저장해준다.
        if string:
            return self.decode_to_string(tokens[0]) #결과값 문장 띄어쓰기 포함해서 만들어준다.
        else:
            return tokens

    def decode_to_string(self, tokens): #디코딩 시에 단어 토큰 뒤에 공백을 넣어준다.
        text = ' '.join(tokens)
        return text.strip() #양쪽의 공백을 없애주고 리턴해준다.

    def cut_eos(self, indices):
        eos_idx = indices.index(self._EOS_ID_)
        return indices[:eos_idx]

    def is_eos(self, voc_id): #id가 EOS_ID??
        return voc_id == self._EOS_ID_

    def is_defined(self, voc_id): #id가 pre_defined 된것  sta_id, eos_id, pad_id, ukw_id 중에 있나?
        return voc_id in self._PRE_DEFINED_

    def max_len(self, batch_set):
        max_len_input = 0
        max_len_output = 0

        for i in range(0, len(batch_set), 2):
            len_input = len(batch_set[i])
            len_output = len(batch_set[i+1])
            if len_input > max_len_input:
                max_len_input = len_input
            if len_output > max_len_output:
                max_len_output = len_output

        return max_len_input, max_len_output + 1

    def pad(self, seq, max_len, start=None, eos=None): # 패딩 해주는 것
        if start: # 디코딩 인풋일 경우
            padded_seq = [self._STA_ID_] + seq
        elif eos: # 타겟 값일 경우
            padded_seq = seq + [self._EOS_ID_]
        else: # 인코딩 인풋일 경우
            padded_seq = seq

        if len(padded_seq) < max_len: # 남은 공간 패딩해준다.
            return padded_seq + ([self._PAD_ID_] * (max_len - len(padded_seq)))
        else:
            return padded_seq

    def pad_left(self, seq, max_len):
        if len(seq) < max_len:
            return ([self._PAD_ID_] * (max_len - len(seq))) + seq
        else:
            return seq

    def stopwordsF(self, sentence):
        sentence = sentence.strip()
        # print(sentence)
        # for i in range(0, len(self.stopwords)):
        #     if sentence.find(self.stopwords[i]) != -1:
        #         print(i, ": ", sentence, "그리고, ", self.stopwords[i])
        # print("for문 끝")

        # for i in range(0, len(self.stopwords)):
        #     # print(a)
        #     if self.stopwords[i] in sentence:
        #         sentence = sentence.replace(self.stopwords[i], "")
        #         print(self.stopwords[i], " 그리고 i : ", i)
        #         # print("SAMES! : ", self.stopwords[i], " and ", sentence)
        #     # else:
        #         # print("NOTSAME! : ", self.stopwords[i], " and ", sentence)
        return sentence
        # print("sentence : ", sentence)
        # print("stopwords : ", self.stopwords[0])
        # print("stopwords : ", self.stopwords[1])
        # print("stopwords : ", self.stopwords[2])
        # print("stopwords : ", self.stopwords[3])


    def transform(self, input, output, input_max, output_max):
        enc_input = self.pad(input, input_max)
        # print("decode input in transform : ", output)
        dec_input = self.pad(output, output_max, start=True)
        target = self.pad(output, output_max, eos=True) #답변의 실제값

        # print(self.vocab_vec)
        # 구글 방식으로 입력을 인코더에 역순으로 입력한다.
        enc_input.reverse()

        #! 시작
        # print("enc_input : ", enc_input,", dec_input : ", dec_input, ", target: ", target)
        enc_list = []
        dec_list = []
        for i in enc_input:
            if i>3:
                enc_list.append(self.vocab_vec[i-4])
            else :
                enc_list.append([float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0)]) #!
        for i in dec_input:
            if i>3:
                dec_list.append(self.vocab_vec[i-4])
            else :
                dec_list.append([float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0),float(0)]) #!

        enc_input = enc_list
        dec_input = dec_list  #! 여기까지
        # enc_input = np.eye(self.vocab_size)[enc_input] #One-hot vector 생성
        # dec_input = np.eye(self.vocab_size)[dec_input] #One-hot vector 생성

        return enc_input, dec_input, target

    def next_batch(self, batch_size): # batch_size만큼의 데이터를 세팅해준다.(한번에 챗봇을 교육시킬 데이터를 세팅해준다.)
        enc_input = []
        dec_input = []
        target = []

        # print("epoc : ", self._index_in_epoch) # _index_in_epoch의 default 값은 1이다.
        start = self._index_in_epoch # 전체 데이터셋에서 교육해야 할 부분 인덱스
        # print("len(self.examples) : ", len(self.examples))
        if self._index_in_epoch + batch_size < len(self.examples) -1:
            self._index_in_epoch += batch_size # 전체 데이터셋에서의 index를 batch_size만큼 더해준다.
        else: # 만약 전체 데이터셋을 교육 시켰다면
            self._index_in_epoch = 1 #! 다시 전체 데이터셋의 1번째 인덱스로 간다.

        # batch_set = self.examples[start:start+batch_size-1]
        temp_batch_set = self.examples[start:start + batch_size] #! examples에서 batch_size만큼 데이터를 가져온다.
        # print("examples : ", self.examples) # [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 11, 2], [...], [...], ...]
        batch_set = [] #! batch_set
        each_batch_set = []
        number = 0
        # print("temp_batch_set : ", temp_batch_set, " <<")
        # print("temp_batch_set[0] : ", temp_batch_set[0]) # [...] << 이거 하나를 가져온다.
        # 작은 데이터셋을 실험하기 위한 꼼수
        # 현재의 답변을 다음 질문의 질문으로 하고, 다음 질문을 답변으로 하여 데이터를 늘린다.
        if FLAGS.data_loop is True:
            # for i in range(0, len(temp_batch_set)): #! 챗봇에 한번에 교육시킬 batch_set을 넣어준다.
            #     # batch_set = batch_set + batch_set[1:] + batch_set[0:1]
            #     batch_set.append(temp_batch_set[i])
            #     batch_set.append(self.examples[0])

            new_start = 0
            for j in range(0, self.numberOfAnswer+1):
                each_temp_batch_set = self.each_examples[new_start:new_start + batch_size]
                new_start += self.each_examples_size[j]  # @ CHANGE는 example만들면서 버렸기 때문에 고려하지 않아도 될 것 같다.
                # print("each_temp_batch_set : ", each_temp_batch_set)
                for i in range(0, len(each_temp_batch_set)):
                    if i+1 < len(each_temp_batch_set):
                        k = i+1
                        each_batch_set.append(each_temp_batch_set[i][1]) #@ 0 2 4 6 8 -> 질문
                        each_batch_set.append(each_temp_batch_set[k][1]) #@ 1 3 5 7 9 -> 대답
                        i+=2
                        # '0' 1 '2' 3 '4' 5 '6' 7 '8' 9 '10'

            # for j in range(0, self.numberOfAnswer + 1):
            #     each_temp_batch_set = self.each_examples[new_start:new_start + batch_size]
            #     new_start += self.each_examples_size[j]  # @ CHANGE는 example만들면서 버렸기 때문에 고려하지 않아도 될 것 같다.
            #     # print("each_temp_batch_set : ", each_temp_batch_set)
            #     for i in range(0, len(each_temp_batch_set)):
            #         each_batch_set.append((each_temp_batch_set[i][1]))
            #         each_batch_set.append(self.each_answer[j])  # @ j번째 대답을 추가한다.

            # print("self.examples[0] : ", self.examples[0])
            # print("batch_set : ", batch_set)
            # print("each_batch_set : ", each_batch_set)
            #0~99, 1~99, 0

        # TODO: 구글처럼 버킷을 이용한 방식으로 변경
        # 간단하게 만들기 위해 구글처럼 버킷을 쓰지 않고 같은 배치는 같은 사이즈를 사용하도록 만듬

        max_len_input, max_len_output = self.max_len(each_batch_set)  # input과 output에 대한 최대 길이를 구해온다. (input은 질문, output은 대답)

        # max_len_input, max_len_output = self.max_len(batch_set) # input과 output에 대한 최대 길이를 구해온다. (input은 질문, output은 대답)

        for i in range(0, len(each_batch_set) - 1, 2):
            enc, dec, tar = self.transform(each_batch_set[i], each_batch_set[i+1],
                                           max_len_input, max_len_output) # encoder, decoder에 넣을 데이터, target값에 들어갈 단어들을 벡터화 해준다.
            # print("enc: ", enc, ",dec: ", dec, ", target: ", target)
            enc_input.append(enc) # enc_input에 정제된 데이터 넣는다.
            dec_input.append(dec) # dec_input에 정제된 데이터 넣는다.
            target.append(tar) # target값에 정제된 데이터 넣는다.

        # print("enc_input : ", enc_input)
        # print("tar : ", target)
        return enc_input, dec_input, target #enc_input, dec_input, target값 리턴

    def tokens_to_ids(self, tokens): # 입력된 문장에서 token화된 단어를 사전이랑 비교하여 새로 리스트를 추가해준다.(id에 token부여)
        ids = []

        for t in tokens: # ['영화', '#', '을', '찾으', '시', '는', ...] 에서 차례로 한 단어씩 t에 넣는다.
            if t in self.vocab_dict:
                ids.append(self.vocab_dict[t]) # vocab_dict에서 그 단어의 인덱스를 찾아서 ids 리스트에 추가한다.
            else:
                ids.append(self._UNK_ID_) # 만약 그 단어가 존재하지 않는다면 에러표시로 숫자 3을 대신 넣는다.
        # print("list ids : ", ids)
        return ids # 최종적으로 ids를 반환한다.

    def ids_to_tokens(self, ids):  # 토큰에 id부여
        tokens = []

        for i in ids:
            tokens.append(self.vocab_list[i])

        return tokens

    def load_examples(self, data_path):
        self.examples = []
        number = 0
        size = 0
        numberOfLine = 0

        with open(data_path, 'r', encoding='utf-8-sig') as content_file: # chat.log 대화 파일 불러옴
            answer = True
            for line in content_file: # 한 줄 씩 읽는다
                if line == 'CHANGE\n':
                    line = content_file.readline() #@ 해당 줄이 CHANGE이 경우 다음 줄로 이동한다!
                    number += 1 # 그 다음 대답과 질문에 관한 정보들이므로 다른 배열에 넣는다.
                    self.each_examples_size.append(size)
                    size = 0
                    answer = True
                elif line == 'END': #@ 파일의 끝임을 확인한다.
                    self.each_examples_size.append(size)
                    size = 0

                tokens = []
                line = self.stopwordsF(line)
                tokens += self.tokenizer(line.strip(), False) # 공백을 지우고 문장을 토큰화한다. #!
                # print("tokens : ", tokens) # tokens : ['영화', '#', '을', '찾으', '시', '는', ...]
                                             # tokens : ['쿠릉쿠', '#', ...]
                ids = self.tokens_to_ids(tokens) # 토큰화된 문장의 단어에 각각 ID 부여하고 리스트 한 줄을 받아온다.
                self.examples.append(ids)  # example에 리스트로 토큰들을 붙인다.

                self.each_examples.append([number, ids])
                if answer == True: #@ 각각의 대답들을 접근하기 쉽게 전역 변수로 설정한다.
                    self.each_answer.append(ids)
                    answer = False

                if self.each_examples[numberOfLine][0] == number: # 각 number의 배열마다 길이를 계산한다.
                    size+=1
                numberOfLine+=1 #@ CHANGE와 END을 제외하고 현재 몇 번째 줄인지 계산한다.

            # print("self.each_examples : ", self.each_examples[2][1])
            # print("each_answer : ", self.each_answer)
            # print("size : ", self.each_examples_size[0])
            # print("size2 : ", self.each_examples_size[1])
            # print("len()", len(self.each_examples))
            self.numberOfAnswer = number

    def tokenizer(self, sentence, vocab_flag): #@
        tw = Twitter()
        str = tw.morphs(sentence.strip())
        analysis = []
        for m in str:
            analysis.append(m)
        self.word_embed_dic.append(analysis)
        return [w for w in analysis if w]

    # def tokenizer(self, sentence, vocab_flag): #!
    #     tw = Twitter() # 트위터 객체를 만들어준다.
    #     # 공백으로 나누고 특수문자는 따로 뽑아낸다.
    #     _TOKEN_RE_ = re.compile("([.,!?\"':;)(☆])") # 특수문자들에 대해서 split할수 있도록 정규식
    #     str = tw.morphs(sentence.strip()) # 트위터 형태소분석기를 통해서 tokenizing
    #     hashtag = "" # 해시태그 변수
    #     hash_flag = False # 해시태그 플래그 (해시태그 키워드인지 아닌지 가려줄 boolean 변수)
    #     analysis = []
    #     for m in str: # str에서 단어 하나씩 loop
    #         if '#' in m: # 만약 단어에 #가 있다면 (키워드라면)
    #             if (len(m) > 1 and m[m.find(',') + 1] == '#' and m.find(',') != -1): # ,#와 같이 특수문자가 2개 같이 있는 경우는 tokenizer에서 붙여서 토큰화하기 때문에 따로
    #                 analysis.append(',') # analysis로 빼준다. analysis에 추가
    #             hash_flag = True # hash_flag를 True로 해준다.
    #         if hash_flag: # hash_flag가 True라면
    #             hashtag += m[m.find('#') + 1:] #! 해시태그된 키워드에서 # 을 뺀 키워드 부분만 저장. # 뒤에 있는 문장들을 저장
    #             # print("hash : ", hashtag)
    #             # hashtag += m[m.find('#'):]
    #             hashtag = hashtag.replace("_", "") # 만약 _가 있다면 없애준다.
    #             analysis.append('#') #! 모델에 학습시킬 문장 부분에는 키워드를 빼고 #만 넣어준다.
    #             # analysis.append(hashtag)
    #             self.keyword.append(hashtag)  # ! keyword에 이전에 #부분을 뺀부분을 넣어준다. (keyword를 넣어줌)
    #             hashtag = "" # 해시태그 변수 비워준다.
    #             hash_flag = False # 해시태그된 키워드를 처리했으니 다시 flag를 False로 바꿔준다.
    #         else:
    #             analysis.append(m) # 트레이닝에 활용될 데이터를 넣어준다.
    #     # print("analysis : ", analysis)
    #     # print("keyword : ", self.keyword) #! 지우면 안됨
    #     # analysis와 동일하기 때문에 words[] 삭제
    #     if (vocab_flag) :
    #         self.word_embed_dic.append(analysis) #!
    #     return [w for w in analysis if w]

    def build_vocab(self, data_path, vocab_path, vector_path, sen_vec_path): #!
        with open(data_path, 'r', encoding='utf-8-sig') as content_file: # 대화파일을 연다. chat.log를 연다.
            words = []
            linetype = True
            for line in content_file:
                if line == 'CHANGE\n':
                    line = content_file.readline()
                elif line == 'END':
                    linetype = False
                if linetype == True:
                    line = self.stopwordsF(line)
                    words += self.tokenizer(line, True) # 문장들을 토큰화한다. # 하나의 배열에 넣어서 길게 만들어진다.
            # print(words)
            words = list(OrderedDict(zip(words, repeat(None)))) # 중복을 없애고 차례대로 단어들을 넣어준다.
            print("words : ", words)

        with open(vocab_path, 'w', encoding='utf-8-sig') as vocab_file: # chat.vog 파일을 만든다.
            for w in words:
                vocab_file.write(w + '\n') # 위에서 정리한 words를 줄 단위로 정리하여 내용을 넣는다.

        self.vocab_list = self._PRE_DEFINED_ + [] # vocab_list가 초기화된다.

        with open(vocab_path, 'r', encoding='utf-8-sig') as vocab_file: # chat.vog을 연다.
            for line in vocab_file: # 한 줄 씩 읽는다.
                self.vocab_list.append(line.strip())  # 미리 지정한 symbol들에 단어장의 단어들을 더한다.
                # vocab_list에는 [0, 1, 2, 3, '영화', '#', '을' ...] 이런식으로 값이 들어가있다.
            # print("vocab list : ", self.vocab_list)

        #! 시작
        # {'_PAD_': 0, '_STA_': 1, '_EOS_': 2, '_UNK_': 3, 'Hello': 4, 'World': 5, ...}
        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)} # 단어를 열거해서 각각 번호를 붙인다.
        # 바로 위에서 정의한 vocab_list에 0번에 0, 1번에 1, 2번에 2, 3번에 3, 4번에 영화, 5번에 ... 등등
        # print("dictpath : ", data_path)
        # print("vocab_dict : ", self.vocab_dict)
        self.vocab_size = len(self.vocab_list) # 단어 사전의 길이
        self.each_vocab_size.append(self.vocab_size) #@ each_vocab_size 리스트에 추가시킨다.
        self.vocab_vec = self.word_embedding() #!  print("vocab_vec[0][0] : ", self.vocab_vec[0][0])

        with open(vector_path, 'w', encoding='utf-8-sig') as vector_file: #! word_embedding.voc을 연다.
            for k, label in enumerate(self.vocab_list[4:]): #!여기까지
        #
                # x, y = self.vocab_vec[i] #!
                a, b, c, d, e, f, g, h, i, j = self.vocab_vec[k] #! -> 한 줄마다 10개의 벡터값이 있으므로 a부터 j까지 그 값을 각각 받아온다.
                # print("vocab_vec : ", self.vocab_vec[k])
                # vector_file.write(str(x) + "," + str(y) + '\n') #! word_embedding파일에 단어들을 벡터화 시킨값들을 넣어준다.
                vector_file.write(str(a) + "," + str(b) + "," + str(c) + "," + str(d) + "," + str(e) + ","
                                  + str(f) + "," + str(g) + "," + str(h) + "," + str(i) + "," + str(j)
                                  + '\n')  # ! word_embedding파일에 단어들을 벡터화 시킨값들을 넣어준다.


        with open(sen_vec_path, 'w', encoding='utf-8-sig') as sen_vec_file: #! word_embedding된 벡터들을 이용해서 sen2vec 적용 sen_embedding.voc을 연다.
            for sen in self.word_embed_dic: # 문장이 모여있는 리스트에서 한 문장씩 갖고옴 (토큰화 되어있음.)
                sen_vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 문장의 벡터를 저장할 변수
                word_count = 0 # 문장 내에 들어가 있는 토큰 개수
                for word in sen: # 한 문장 내에 토큰화 된 단어를 하나씩 꺼내온다.
                    # print("word: ", word)
                    sen_vec += self.vocab_vec[self.vocab_dict.get(word) - 4] # 해당 단어에 대한 벡터를 더해준다.
                    # print("word_vec : ", self.vocab_vec[self.vocab_dict.get(word)-4])
                    word_count += 1 # 단어 수만큼 더함
                # print("before sen_vec : ", sen_vec)
                sen_vec = sen_vec / word_count # 단어 벡터의 합의 평균을 구해준다.
                # print("word_count : ", word_count)
                # print("sen_vec : ", sen_vec)
                sen_vec_file.write(str(sen_vec[0]) + "," + str(sen_vec[1]) + "," + str(sen_vec[2]) + "," + str(sen_vec[3]) + "," + str(sen_vec[4]) + ","
                                  + str(sen_vec[5]) + "," + str(sen_vec[6]) + "," + str(sen_vec[7]) + "," + str(sen_vec[8]) + "," + str(sen_vec[9])
                                  + '\n') # 그리고 그 값을 파일에 써준다.

    def load_vocab(self, vocab_path, vector_path):
        self.vocab_list = self._PRE_DEFINED_ + [] # 미리 defined된 symbol들에 단어장의 단어들을 더한 list
        # print("vocab_list : ", self.vocab_list) -> [0, 1, 2, 3]

        with open(vocab_path, 'r', encoding='utf-8-sig') as vocab_file: # chat.vog을 연다.
            for line in vocab_file: # 한 줄 씩 읽는다.
                self.vocab_list.append(line.strip())
            # print(self.vocab_list) -> [0, 1, 2, 3, '영화', '#', '을', '찾으'. '시', ...]

        # {'_PAD_': 0, '_STA_': 1, '_EOS_': 2, '_UNK_': 3, 'Hello': 4, 'World': 5, ...}
        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)} # 단어를 열거해서 각각 번호를 붙인다.
        self.vocab_size = len(self.vocab_list) # 단어 사전의 길이

        #! 시작
        with open(vector_path, 'r', encoding='utf-8-sig') as vec_file: # word_embbeding.voc을 연다. 단어들이 벡터화 돼있는 단어장
            for line in vec_file: # 한 줄 씩 읽는다.
                # print("line : ", line) -> 0.51984763,-0.14020371,0.36940932,-0.3891697,0.53203845,0.7021496,-0.99700403,-0.2890625,0.8061869,0.6083009 이게 한 줄
                list = [float(i) for i in line.strip().split(',')] # 이거 자체가 for문
                # print("list : ", list) -> [0.51984763, -0.14020371, 0.36940932, -0.3891697, 0.53203845, 0.7021496, -0.99700403, -0.2890625, 0.8061869, 0.6083009] 이게 한 줄
                self.vocab_vec.append(list) #! 여기까지

        with open(FLAGS.sen_vec_path, 'r', encoding='utf-8-sig') as sen_vec_file: # sen_embedding.voc을 연다. 위와 똑같은 진행
            for line in sen_vec_file:  # 한 줄 씩 읽는다.
                # print(line)
                list = [float(i) for i in line.strip().split(',')]
                self.sentence_vec.append(list)

    # skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를 생성하는 함수
    def random_batch(self, data, size): #! 시작
        random_inputs = []
        random_labels = []
        random_index = np.random.choice(range(len(data)), size, replace=False)
        # size 개수만큼의 배열을 0~ (len(data)-1) 까지의 수중에 골라서 만든다.(중복허용 X == (replace=False))

        for i in random_index:
            random_inputs.append(data[i][0])  # target 게임, 게임 ...
            random_labels.append([data[i][1]])  # context word 나, 만화...

        return random_inputs, random_labels #! 여기까지

    def word_embedding(self): #! 시작
        print("word_embed_dic : ", self.word_embed_dic)
        for list in self.word_embed_dic: # [['영화', '#', '을', '찾으', '시', '는', '군요', '.', '검색', '하겠', '습니다', '.'], [...], [...]...]] 이런식으로 구성되는데
                                          # [...] 하나를 list로 대입시킨다.
            # print(list)
            vocab_index = [self.vocab_dict[word] for word in list] #! build_vocab에서 dictionary 형태로 정의했던 vocab_dict 사용
            # print(vocab_index)
            # with open(data_path, 'r', encoding='utf-8-sig') as content_file: #로그파일 불러옴
            #     for line in content_file: #한줄 씩 읽는다
            # vocab_index = [self.]
            # print(vocab_index)
            # ! 단어에 대한 인덱스를 저장해줌.
            if(len(list) < 2):
                pass
            elif(len(list) == 2):
                self.skip_grams.append([vocab_index[0]-4, vocab_index[1]-4])
                self.skip_grams.append([vocab_index[1]-4, vocab_index[0]-4])
            else:
                for i in range(1, len(list) - 1):  # 2번째 인덱스부터 끝에서 앞에 단어까지
                    # (context, target) : ([target index - 1, target index + 1], target)
                    target = vocab_index[i]-4 #!
                    # print("target : ", target)
                    context = [vocab_index[i - 1]-4, vocab_index[i + 1]-4] #!
                    # print("context : ", context)
                    # "나 게임 만화 애니"에서 target이 게임이 된다면 context에는 [나, 만화]가 들어가게 된다.

                    # (target, context[0]), (target, context[1])..
                    for w in context:
                        self.skip_grams.append([target, w])  # skip-gram에 [게임, 나], [게임, 만화]가 들어가게 된다.
        # print(self.skip_grams)
        inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        # tf.nn.nce_loss 를 사용하려면 출력값을 이렇게 [FLAGS.batch_size, 1] 구성해야합니다.
        labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])

        # word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수입니다.
        # 총 단어 갯수와 임베딩 갯수를 크기로 하는 두 개의 차원을 갖습니다.
        embeddings = tf.Variable(tf.random_uniform([self.vocab_size-4, FLAGS.embedding_size], -1.0, 1.0))
        # 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
        # 예) embeddings     inputs    selected
        #    [[1, 2, 3]  -> [1, 2] -> [[2, 3, 4]
        #     [2, 3, 4]                [3, 4, 5]]
        #     [3, 4, 5]
        #     [4, 5, 6]]
        selected_embed = tf.nn.embedding_lookup(embeddings, inputs)  ##########
        # nce_loss 함수에서 사용할 변수들을 정의합니다.
        nce_weights = tf.Variable(tf.random_uniform([self.vocab_size-4, FLAGS.embedding_size], -1.0, 1.0)) # weight을 -1과 1사이에 값을 준다.
        # nce_biases = tf.Variable(tf.zeros([self.vocab_size-4]))
        nce_biases = tf.Variable(tf.random_normal([self.vocab_size-4])) #! bias 설정.

        # nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
        # 함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하기만 하면 됩니다.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, FLAGS.num_sampled, self.vocab_size-4))
        train_op = tf.train.AdamOptimizer(0.1).minimize(loss) # adamOptimizer를 통해서 loss를 최소화...

        #########
        # 신경망 모델 학습
        #########

        with tf.Session() as sess:
            init = tf.global_variables_initializer() #변수 초기화
            sess.run(init)

            for step in range(1, FLAGS.epoch + 1):
                batch_inputs, batch_labels = self.random_batch(self.skip_grams, FLAGS.batch_size) #! skipgram적용한 모델과 batchsize만큼 랜덤으로 배치한다.
                # print("batch_inputs: ", batch_inputs, ", batch_labels: ", batch_labels)

                _, loss_val = sess.run([train_op, loss],
                                       feed_dict={inputs: batch_inputs,
                                                  labels: batch_labels}) #세션 실행해서 loss값을 얻어낼 수 있다.

                # if step % 10 == 0:
                #     print("loss at step ", step, ": ", loss_val)

            # matplot 으로 출력하여 시각적으로 확인해보기 위해
            # 임베딩 벡터의 결과 값을 계산하여 저장합니다.
            # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있습니다.
            trained_embeddings = embeddings.eval()

        # print("trained", trained_embeddings)
        # print("traind len : " + str(len(trained_embeddings)))
        # print("word list len : " + str(len(self.vocab_list[4:])))

        # for i, label in enumerate(self.vocab_list[4:]):
        #     x, y = trained_embeddings[i]
        #     # print(trained_embeddings[i]," : ", label)
        #     plt.scatter(x, y)
        #     plt.annotate(label, xy=(x, y), xytext=(5, 2),
        #                  textcoords='offset points', ha='right', va='bottom')
        # plt.show()

        # print("tr : ", trained_embeddings)
        return trained_embeddings #! 여기까지

def main(_):
    dialog = Dialog() #dialog 객체 생성

    if FLAGS.data_path and FLAGS.voc_test: # chat.log 파일 존재하고 voc_test가 True일 경우
        print("다음 데이터로 어휘 사전을 테스트합니다.", FLAGS.data_path)
        dialog.load_vocab(FLAGS.voc_path, FLAGS.vec_path) #! 단어사전 로드
        dialog.load_examples(FLAGS.data_path)  # 단어 리스트 로드

        enc, dec, target = dialog.next_batch(10) # 10개의 data만 테스트
        # print(target) # target 출력
        enc, dec, target = dialog.next_batch(10) # 10개의 data만 테스트
        # print(target) # target 출력

    elif FLAGS.data_path and FLAGS.voc_build: # chat.log 파일 존재하고 voc_build가 True일 경우
        print("다음 데이터에서 어휘 사전을 생성합니다.")
        print("첫 번째 사전 생성 ", FLAGS.data_path)
        dialog.build_vocab(FLAGS.data_path, FLAGS.voc_path, FLAGS.vec_path, FLAGS.sen_vec_path)  #! 단어사전 만들기

        # if FLAGS.data_path2 and FLAGS.voc_build:
        #     dialog.word_embed_dic.clear()
        #     dialog.__init__()
        #     print("다음 데이터에서 어휘 사전을 생성합니다.")
        #     print("두 번째 사전 생성 ", FLAGS.data_path2)
        #     dialog.build_vocab(FLAGS.data_path2, FLAGS.voc_path2, FLAGS.vec_path2, FLAGS.sen_vec_path2)  #! 단어사전 만들기

    elif FLAGS.voc_test:
        dialog.load_vocab(FLAGS.voc_path, FLAGS.vec_path) #! 사전 데이터 load
        print("사전출력 : ", dialog.vocab_dict) # 사전출력


if __name__ == "__main__":
    tf.app.run() #어플 실행
