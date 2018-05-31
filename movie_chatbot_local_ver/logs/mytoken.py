from konlpy.tag import Mecab  # hashtag 얻어올라고

def tokenizer(iterator):
    # tw = Twitter()
    tw = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    a = []
    for x in iterator:
    	a.append(tw.morphs(x.strip()))
    return a
    #return (tw.morphs(x.strip()) for x in iterator)
