import jieba
# import word2vec
from gensim.models import word2vec
import numpy as np
from corpus_reader import TmxHandler
import tokenizer as tknz
from tokenizer import Tokenizer

def generate_train_file(fname, ofname):
    
    eos_en = tknz.EOS_EN
    eos_zh = tknz.EOS_ZH
    reader = TmxHandler()
    tokenizer = Tokenizer()
    r = reader.parse(fname)
    n_lines = 0
    max_lines = 50000
    chunk_idx = 0
    finished = False

    while not finished:
        with open(ofname, 'a', encoding='utf-8') as ff:
            while True:
                try:
                    corpus = next(r)
                    if 'en' in corpus:
                        scent = corpus['en'] + eos_en
                        ws = tokenizer.tokenize(scent, 'en')
                        ff.write(' '.join(ws) + '\n') # 词汇用空格分开
                        # n_lines += 1
                    if 'zh' in corpus:
                        scent = corpus['zh'] + eos_zh
                        ws = tokenizer.tokenize(scent, 'zh')
                        ff.write(' '.join(ws) + '\n') # 词汇用空格分开
                        # n_lines += 1
                    corpus = None
                    # if n_lines >= max_lines:
                    #     chunk_idx += 1
                    #     n_lines = 0
                    #     break
                except StopIteration:
                    finished = True
                    break

def generate_vocab_from_model(model_fname, ofname):
    model = word2vec.Word2Vec()
    print("load embedding model")
    model = model.wv.load_word2vec_format(model_fname)
    print("load embedding model finished. Start building old vocabulary.")
    with open(ofname, 'a', encoding='utf-8') as ff:
        for w in model.key_to_index.keys():
            ff.write(w + '\n')

def add_eos(in_dir, ofname):
    eos_en = tknz.EOS_EN
    eos_zh = tknz.EOS_ZH

    train_input = word2vec.PathLineSentences(in_dir)
    iter = train_input.__iter__()

    with open(ofname, 'a', encoding='utf-8') as ff:
        for l in iter:
            l = next(iter)
            l.append(eos_zh)
            ff.write(' '.join(l) + '\n')

def save_vectors(model_fname, out_vector_fname):
    print("loading model")
    model = word2vec.Word2Vec.load(model_fname)
    print("saving embeddings")
    model.wv.save_word2vec_format(out_vector_fname)

def train_incremental(model_fname, train_dir, out_vector_fname, out_model_fname):
    train_input = word2vec.PathLineSentences(train_dir)
    # train_input = word2vec.LineSentences(train_dir)
    model = word2vec.Word2Vec.load(model_fname)

    print("Loading new vocab")
    model.build_vocab(train_input, update=True)

    print("Start training.")
    model.train(train_input, total_examples=model.corpus_count, epochs=model.epochs)

    print("saving new model")
    model.save(out_model_fname)

    print("saving vectors.")
    model.sv.save_word2vec_format(out_vector_fname)    

def train(model_fname, train_fname, out_fname, out_model_fname):
    eos_en = tknz.EOS_EN
    eos_zh = tknz.EOS_ZH

    # train_input = word2vec.LineSentence(train_fname)
    train_input = word2vec.PathLineSentences(train_fname)
    model = word2vec.Word2Vec(min_count=1, vector_size=300, workers=3, epochs=20)

    print("building training vocab")
    model.build_vocab(train_input)
    model.wv.vectors_lockf = np.ones(len(model.wv))

    # print("intersecting old model")
    # model.wv.intersect_word2vec_format(model_fname)
    # print("load embedding model")
    # model.wv.load_word2vec_format(model_fname)
    # print("load embedding model finished. Start updating vocabulary.")
    print("Start training.")
    model.train(train_input, total_examples=model.corpus_count, epochs=model.epochs)

    print("saving new model")
    model.save(out_model_fname)

    print("loading old embeddings")
    model_old = word2vec.Word2Vec(vector_size=300, min_count=1)
    # vocab_input = word2vec.LineSentence(vocab_fname)
    # model_old.build_vocab(vocab_input)
    model_old_kv = model_old.wv.load_word2vec_format(model_fname)

    print("adding eos embeddings")
    eos_en_embedding = model.wv.get_vector(eos_en)
    eos_zh_embedding = model.wv.get_vector(eos_zh)
    model_old_kv.add_vector(eos_en, eos_en_embedding)
    model_old_kv.add_vector(eos_zh, eos_zh_embedding)

    print("saving updated model.")
    model_old_kv.save_word2vec_format(out_fname)

from nltk.corpus import brown as b
import nltk

def download_brown(dir_name):
    print(b.categories())# 打印分类名
    for file in b.fileids():
        file_text = b.words(file) #获取分词后的文本
        with open(dir_name + "/" + file,"a+",encoding="utf-8") as f:#拼接列表
            f.write(" ".join(file_text))

        
if __name__=="__main__":
    
    # nltk.download('brown')
    # add_eos('../corpus/tokenized/OPENSLR-SLR55-CLMAD', '../corpus/tokenized/eos/clmad.txt')
    # generate_vocab_from_model("../sgns.merge.word", "../sgns.merge.word.vocab")
    #generate_train_file("../corpus/CCMatrix_v1.tmx_bk","../corpus/tokenized/eos/CCMatrix_v1_tokenized.txt")
    # train("../sgns.merge.word", "../corpus/tokenized/eos", "../sgns.merge.word.eos", "../parellel_01.model")
    # save_vectors("../parellel_01.model", "../parellel_01.v2c")
    train_incremental("../embeddings/parellel_01.model", "../corpus/tokenized/embedding_eos", "../embeddings/parellel_02.v2c", "../embeddings/parellel_02.model")
