import tensorflow as tf
import random
import math
import os
import numpy as np
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from Rnn_chatbot.config import FLAGS
from Rnn_chatbot.model import Seq2Seq
from Rnn_chatbot.dialog import Dialog

def train(dialog, batch_size, epoch):
    tf.reset_default_graph()
    model = Seq2Seq(dialog.vocab_size)
    # model = Seq2Seq(200)

    # print("size : ", dialog.vocab_size)

    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir) # checkpoint 얻는다.(모델의 Variable값을 얻어옴)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): # 모델 checkpoint가 존재하면
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path) # checkpoint파일에서 모델의 변수값을 얻어온다.
        else: # 모델 checkpoint가 존재하지 않는다면
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer()) # Variable 모두 초기화하고 세션을 실행시킨다.

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph) # 트레이닝 로그를 남기기위해 파일 Writer를 연다.

        # print("dialog.examples : ", len(dialog.examples))

        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))
        # total_batch * batch_size = examples의 갯수(training시켜야할 단어 개수)

        for step in range(total_batch * epoch):
            enc_input, dec_input, targets = dialog.next_batch(batch_size) # batch_size만큼 데이터를 신경망에 넣어주고, 다음 train을 위해 세팅해준다.

            _, loss = model.train(sess, enc_input, dec_input, targets) # 트레이닝 (세션 실행)

            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets) # epoch 100번째 마다 로그 써줌

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss))

        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name) # 트레이닝 끝나면 체크포인트 적어줌
        model.saver.save(sess, checkpoint_path, global_step=model.global_step) # 변수저장 체크포인트 저장

    print('최적화 완료!')

def clearTrain():
    print("hi")


# def test(dialog, batch_size=100):
#     print("\n=== 예측 테스트 ===")
#
#     model = Seq2Seq(dialog.vocab_size)
#
#     with tf.Session() as sess:
#         ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
#         print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
#         model.saver.restore(sess, ckpt.model_checkpoint_path)
#
#         enc_input, dec_input, targets = dialog.next_batch(batch_size)
#
#         expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)
#
#         expect = dialog.decode(expect)
#         outputs = dialog.decode(outputs)
#
#         pick = random.randrange(0, len(expect) / 2)
#         input = dialog.decode([dialog.examples[pick * 2]], True)
#         expect = dialog.decode([dialog.examples[pick * 2 + 1]], True)
#         outputs = dialog.cut_eos(outputs[pick])
#
#         print("\n정확도:", accuracy)
#         print("랜덤 결과\n")
#         print("    입력값:", input)
#         print("    실제값:", expect)
#         print("    예측값:", ' '.join(outputs))


def main(_):
    dialog = Dialog() # Dialog 객체 생성
    dialog.load_vocab(FLAGS.voc_path, FLAGS.vec_path) # 단어 불러오기 # chat.voc과 word_embedding.voc을 인자로 넣는다.
                                                      # vocab_dictd와 "vocab_size"에 대한 정보를 가져온다.
    dialog.load_examples(FLAGS.data_path) # example에 문장들을 토큰화해서 리스트로 붙인다.

    # dialog.load_vocab(FLAGS.voc_path2, FLAGS.vec_path2)  # 단어 불러오기 # chat.voc과 word_embedding.voc을 인자로 넣는다.
    # #                                                       # vocab_dict와 "vocab_size"에 대한 정보를 가져온다.
    # dialog.load_examples(FLAGS.data_path2)  # example에 문장들을 토큰화해서 리스트로 붙인다.

    if FLAGS.train: # train이 True일 경우
        train(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
        # train(dialog2, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
    # elif FLAGS.test:
    #     test(dialog, batch_size=FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()
