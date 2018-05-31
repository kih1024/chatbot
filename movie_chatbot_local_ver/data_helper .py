import re
import numpy as np
import pandas as pd
from konlpy.tag import Mecab
from tensorflow.contrib.learn import preprocessing

labels = ["actor", "release_time", "total_attendance", "running_time", "movie_title", "movie_grade"] #%

actor = ["배우", "영화배우", "탤런트", "탈렌트", "베우", "액터", "엑터", "텔랜트", "탤렌트", "탈랜트", "출연", "연기", "qodn"] #%
release_time = ["개봉", "개봉일", "개봉날", "시작", "언제", "roqhd", "djswp", "skdhs", "나왔"] #%
total_attendance = ["누적", "관객", "관객수", "snwjr", "rhksror"] #%
running_time = ["러닝", "타임", "상영", "시간", "tlrks", "fjsld"] #%
movie_title = ["영화", "제목", "타이틀", "wpahr", "dudghk"] #%
movie_grade = ["평점", "평", "별", "별점", "점수", "vud"] #%

keywords = [actor, release_time, total_attendance, running_time, movie_title, movie_grade] #%


def clean_str(s):  # by taeuk
    # 정규표현식을 이용하여 문자열 정리
    sub_list = [r"ㄱ",r"ㄴ",r"ㄷ",r"ㄹ",r"ㅁ",r"ㅂ",r"ㅅ",r"ㅇ",r"ㅈ",r"ㅊ",r"ㅋ",r"ㅌ",r"ㅍ",r"ㅎ",
        r"ㅄ",r"ㄾ",r"ㄺ",r"ㅀ",r"ㄶ",r"ㄳ",r"ㅀ",r"ㄶ",r"ㄳ",r"ㄿ",r"ㄲ",r"ㄸ",r"ㅃ",r"ㅆ",r"ㅉ",
        r"ㅡ",r"ㅠ",r"ㅜ",r"^"]
    for i in sub_list:
        s = re.sub(i,"",s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\?", " \? ", s)
    return s.strip()


def load_data_and_labels(filename): 
    """Load sentences and labels"""
    df = pd.read_csv(filename, compression='zip', dtype={'데이터셋': object}, engine='python',encoding='CP949')  # movie_dataset.csv.zip 읽기
    selected = ['클래스분류', '데이터셋']  # 파일 중 가져올 열을 지정.
    # non_selected = list(set(df.columns) - set(selected))  # 선택 안된놈들 -> 어차피 필요한 열 빼고 다 삭제할거니까 이 변수자체가 필요없어질듯!!
    # df = df.drop(non_selected, axis=1)  # Drop non selected columns -> non_selected 없애도 될거같으니 이 문장 또한 없어질듯!!
    df = df.dropna(axis=0, how='any', subset=selected)  # 비어있는 행 제거
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe

    labels = sorted(list(set(df[selected[0]].tolist())))  # 분류되는 6가지 라벨
    one_hot = np.zeros((len(labels), len(labels)), int)  # 라벨 원핫 인코딩
    np.fill_diagonal(one_hot, 1) # 라벨에 대해 원핫 벡터를 생성
    label_dict = dict(zip(labels, one_hot))

    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()  # 라벨에 대한 원핫 벡터 들어감

    return x_raw, y_raw, df, labels


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def build_vocab(max_len, data_set, num_of_class, Flag):
    vocab_dict = preprocessing.CategoricalVocabulary()
    max_num = max_len
    temp_Flag = False

    if Flag:
        with open("./data/word_list.voc", 'w', encoding='utf-8-sig') as vocab_file:  # 단어장을 연다.
            vocab_file.write(str(max_num) + '\n')
            for i, w in enumerate(data_set):
                vocab_dict.add(w, i+1)
                vocab_file.write(w + '\n')
        with open("./data/keyword_vec.vec", 'w', encoding='utf-8-sig') as vec_file:  # 단어장을 연다.
            for _, w in enumerate(data_set):
                temp_Flag = False
                for i in range(len(keywords)):
                    for j in keywords[i]:
                        if j in w:
                            vec_file.write(str(i) + '\n')
                            temp_Flag = True
                            break
                    if(temp_Flag):
                        break
                if temp_Flag == False:
                    vec_file.write(str(num_of_class) + '\n')

    else:
        with open("./data/word_list.voc", 'r', encoding='utf-8-sig') as vocab_file:  # 단어장을 연다.
            i=0
            for line in vocab_file:
                if i==0:
                    max_num = int(line.strip())
                    i += 1
                else:
                    vocab_dict.add(line.strip(),i)
                    i += 1

    vocab_dict.freeze(True)

    return vocab_dict, max_num


def tokenizer(sentence, vocab_flag):
    tw = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic') # 트위터 객체를 만들어준다.

    # 공백으로 나누고 특수문자는 따로 뽑아낸다.
    temp = sentence.split()
    keyword = []
    x = sentence
    for key in temp:
        if '#' in key:
            keyword.insert(0, key[1:].replace('_',"").replace('#',""))
            x = x.replace(key, '#')
    words = []
    words.append(tw.morphs(x.strip()))  # 트위터 형태소분석기를 통해서 tokenizing
    return [w for w in words if w]


if __name__ == '__main__':
    # data_helper의 main은 필요없는듯. by odg
    input_file = './data/movie_dataset.csv.zip'
    load_data_and_labels(input_file)
