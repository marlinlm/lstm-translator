import jieba
from nltk import word_tokenize as wt
from nltk import MWETokenizer

EOS = '<eos>'

class Tokenizer:
    def __init__(self):
        self.tokenizer = MWETokenizer([('<','eos','>'),('<','unk','>'),('<','pad','>')],separator="")

    def tokenize(self, s, lang):
        if(lang == 'zh'):
            return self.tokenizer.tokenize(jieba.lcut(s, cut_all=False))
        elif lang == 'en':
            return self.tokenizer.tokenize((wt(s)))
        else:
            raise Exception('Unsupported language ' + lang)

if __name__=="__main__":
    tokenizer = Tokenizer()
    print(tokenizer.tokenize("I <unk> football! <eos><pad><pad><pad><pad>", 'en'))
    print(tokenizer.tokenize("我爱<unk>足球。<eos><pad><pad><pad><pad>", 'zh'))


        