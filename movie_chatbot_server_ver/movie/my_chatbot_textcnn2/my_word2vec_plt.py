from sklearn.manifold import TSNE
import re
import matplotlib
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import pandas as pd

# 참고한 사이트 https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne/code

font_name = matplotlib.font_manager.FontProperties(
                fname="C:/Windows/Fonts/.ttf"  # 한글 폰트 위치를 넣어주세요
            ).get_name()
matplotlib.rc('font', family=font_name)

path_one="/Users/lemon/Desktop/multi-class-text-classification-cnn-master_combine/"
path_model="trained_model_1526470663"
path_two="/word2Vec.vec"

modelPath=path_one+path_model+path_two
# C:\Users\lemon\Desktop\multi-class-text-classification-cnn-master_combine\trained_model_1526302044
# C:\Users\lemon\Desktop\multi-class-text-classification-cnn-master_combine\trained_model_1526302044/word2Vec.vec
# C:/Users/lemon/Desktop/multi-class-text-classification-cnn-master_combine/trained_model_1526302044/word2Vec.vec
# /Users/lemon/Desktop/multi-class-text-classification-cnn-master_combine/trained_model_1526302044/word2Vec.vec

# model = g.Doc2Vec.load(modelPath)
model = Word2Vec.load(modelPath)

vocab = list(model.wv.vocab)
X = model[vocab]
# X = model[model.wv.vocab]
tsne = TSNE(n_components=2)
# X_tsne = tsne.fit_transform(X[:1000,:])
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
plt.show()

