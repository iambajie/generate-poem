import numpy as np
import word2vec

class Word2Vec():
    def __init__(self,file_path):
        self.model = word2vec.load(file_path)

        self.add_word('<unknown>')
        self.add_word('<pad>')

    def add_word(self,word):
        #在哈希链表中查找 词位置
        if word not in self.model.vocab_hash:
            w_vec = np.random.uniform(-0.1,0.1,size=128)
            self.model.vocab_hash[word] = len(self.model.vocab)
            self.model.vectors = np.row_stack((self.model.vectors,w_vec))
            self.model.vocab = np.concatenate((self.model.vocab,np.array([word])))



if __name__ == '__main__':
    w2vpath = './data/poem/vectors_poem.bin' #分词

    w2v = Word2Vec(w2vpath)
    with open( './data/poem/vectors_poem.txt','w',encoding='utf-8') as fw:
        for w in w2v.model.vocab:
            fw.writelines(w + '\n')
