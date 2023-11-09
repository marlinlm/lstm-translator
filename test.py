import jieba
from nltk import word_tokenize as wt
from nltk import MWETokenizer

# jieba.load_userdict(['停动烫次烫打烫次停'])
# tokens = jieba.lcut("我要去北京旅游。停动烫次烫打烫次停", cut_all=False)
# print (len(tokens))
# print(tokens)

# sen = 'i love football players.<eos>/r/n'
# user_tokenizer = MWETokenizer([('<','eos','>')])
# w = user_tokenizer.tokenize((wt(sen)))
# print(w)

import word2vec
from gensim.models import Word2Vec

model = Word2Vec.load("../parellel_01.model")
print(model['<eos>'])

# w2v_eos = word2vec.WordEmbeddingLoader("../sgns.merge.word.eos")
# print(w2v_eos.get_embeddings('是'))

# w2v = word2vec.WordEmbeddingLoader("../baike_26g_news_13g_novel_229g.model")
# w2v.get_embeddings('是')