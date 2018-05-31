import tensorflow as tf


tf.app.flags.DEFINE_string("train_dir", "./Rnn_chatbot/model", "학습한 신경망을 저장할 폴더")
tf.app.flags.DEFINE_string("log_dir", "./Rnn_chatbot/logs", "로그를 저장할 폴더")
tf.app.flags.DEFINE_string("ckpt_name", "conversation.ckpt", "체크포인트 파일명")

tf.app.flags.DEFINE_boolean("train", True, "학습을 진행합니다.")
tf.app.flags.DEFINE_boolean("test", False, "테스트를 합니다.") ## 실행시 명령인자로 줄 수 있다.
tf.app.flags.DEFINE_boolean("data_loop", True, "작은 데이터셋을 실험해보기 위해 사용합니다.")
tf.app.flags.DEFINE_integer("batch_size", 100, "미니 배치 크기")
tf.app.flags.DEFINE_integer("epoch", 600, "총 학습 반복 횟수")
tf.app.flags.DEFINE_integer("embedding_size", 10, "단어 벡터를 구성할 임베딩 차원의 크기") #!
tf.app.flags.DEFINE_integer("num_sampled", 10, "word2vec 모델을 학습시키기위한 nce_loss 함수에서 사용하기 위한 샘플링 크기 #batch_size보다 작아야함") #!

tf.app.flags.DEFINE_string("data_path", "./Rnn_chatbot/data/chat.log", "대화 파일 위치")
tf.app.flags.DEFINE_string("voc_path", "./Rnn_chatbot/data/chat.voc", "어휘 사전 파일 위치")
tf.app.flags.DEFINE_string("vec_path", "./Rnn_chatbot/data/word_embedding.voc", "단어 워드 임베딩 사전 파일 위치") #!
tf.app.flags.DEFINE_string("sen_vec_path", "./Rnn_chatbot/data/sen_embedding.voc", "문장 임베딩 사전 파일 위치") #! 대화 파일 문장 순서와 동일

tf.app.flags.DEFINE_string("stopwords", "./Rnn_chatbot/stopwords/stopwords.txt", "제외할 단어 텍스트 파일")

# tf.app.flags.DEFINE_string("data_path2", "./data/chat3.log", "대화 파일 위치")
# tf.app.flags.DEFINE_string("voc_path2", "./data/chat3.voc", "어휘 사전 파일 위치")
# tf.app.flags.DEFINE_string("vec_path2", "./data/word_embedding3.voc", "단어 워드 임베딩 사전 파일 위치") #!
# tf.app.flags.DEFINE_string("sen_vec_path2", "./data/sen_embedding3.voc", "문장 임베딩 사전 파일 위치") #! 대화 파일 문장 순서와 동일

tf.app.flags.DEFINE_boolean("voc_test", False, "어휘 사전을 테스트합니다.")
tf.app.flags.DEFINE_boolean("voc_build", True, "주어진 대화 파일을 이용해 어휘 사전을 작성합니다.")
tf.app.flags.DEFINE_integer("max_decode_len", 20, "최대 디코더 셀 크기 = 최대 답변 크기.")

FLAGS = tf.app.flags.FLAGS