import jieba
from nltk import word_tokenize as wt
from nltk import MWETokenizer

EOS_ZH = '停动烫次烫打烫次停'
EOS_EN = '<eos>'

class Tokenizer:
    def __init__(self):
        jieba.load_userdict([EOS_ZH])
        self.en_tokenizer = MWETokenizer([('<','eos','>')],separator="")

    def tokenize(self, s, lang):
        if(lang == 'zh'):
            return jieba.lcut(s, cut_all=False)
        elif lang == 'en':
            return self.en_tokenizer.tokenize((wt(s)))
        else:
            raise Exception('Unsupported language ' + lang)

if __name__=="__main__":
    tokenizer = Tokenizer()
    print(tokenizer.tokenize("I love football! <eos>", 'en'))
        