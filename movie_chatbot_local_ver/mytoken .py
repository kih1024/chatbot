from konlpy.tag import Mecab  # hashtag 얻어올라고

def tokenizer(iterator):
    # tw = Twitter()
    tw = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    a = []
    for x in iterator:
        sentence = x
        temp = x.split()
        for key in temp:
            if '#' in key:
               sentence = sentence.replace(key, "#")
        a.append(tw.morphs(sentence.strip()))
    return a
    #return (tw.morphs(x.strip()) for x in iterator)
