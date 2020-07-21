import json
import os, sys,time
import logging
import math
import numpy as np
import tensorflow as tf
from poem_charrnn import CharRNNLM,SampleType
from poem_config import config_sample
from poem_word2vec import Word2Vec


class  PoemWrite():
    def __init__(self,args):
        self.args = args

        #打印日志时间、级别、信息
        logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s:%(message)s',
                        level=logging.INFO, datefmt='%I:%M:%S')
        path=os.path.join(self.args.model_dir, 'result.json')
        with open(os.path.join(self.args.model_dir, 'result.json'), 'r') as f:
            result = json.load(f)

        params = result['params']
        best_model = result['best_model']
        best_valid_ppl = result['best_valid_ppl']
        if 'encoding' in result:
            self.args.encoding = result['encoding']
        else:
            self.args.encoding = 'utf-8'

        base_path = args.data_dir
        w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.w2v = Word2Vec(w2v_file)
        if args.seed >= 0:
            np.random.seed(args.seed)

        logging.info('best_model: %s\n', best_model)

        self.sess = tf.Session()
        w2v_vocab_size = len(self.w2v.model.vocab)
        with tf.name_scope('evaluation'):
            self.model = CharRNNLM(is_training=False,w2v_model = self.w2v.model,vocab_size=w2v_vocab_size, infer=True, **params)
            saver = tf.train.Saver(name='model_saver')
            saver.restore(self.sess, best_model)

    def free_verse(self,num_sentence):
        sample = self.model.sample_seq(self.sess, 60, '[')
        if not sample:
            return 'err occar!'

        print('free_verse:',sample)

        idx_end = sample.find(']')
        parts = sample.split('。')

        #控制写几句诗
        sentence_len=0
        m=0
        num_sentence=int(num_sentence)
        if(len(parts)>=num_sentence):
            m=num_sentence
        else:
            m=len(parts)
        for i in range(m):
            sentence_len += len(parts[i])

        if sentence_len<idx_end:
            return sample[1:sentence_len + m]
        else:
            n=len(sample[1:idx_end].split('。'))
            additional='（对不起，现在只能想到{}句诗）'.format(n-1)
            return sample[1:idx_end]+additional


    def cangtou(self,given_text):
        #根据给定的文字写藏头诗
        if(not given_text):
            return self.rhyme_verse()

        start = ''

        for i,word in enumerate(given_text):
            word = ''
            if i < len(given_text):
                word = given_text[i]

            if i == 0:
                start = '[' + word
            else:
                start += word

            before_idx = len(start)
            sample = self.model.sample_seq(self.sess, self.args.length, start)
            print('Sampled text is:\n\n%s' % sample)

            sample = sample[before_idx:]
            idx1 = sample.find('，')
            idx2 = sample.find('。')
            min_idx = min(idx1,idx2)

            if min_idx == -1:
                if idx1 > -1 :
                    min_idx = idx1
                else: min_idx =idx2
            if min_idx > 0:
                start ='{}{}'.format(start, sample[:min_idx + 1])

            print('last_sample text is:\n\n%s' % start)

        return start[1:]

def start_model():
    now = int(time.time())
    args = config_sample('--model_dir output_poem --length 20 --seed {}'.format(now))
    writer = PoemWrite(args)
    return writer

if __name__ == '__main__':
    writer = start_model()
    text=[]
    # text=writer.free_verse(2)
    text=writer.cangtou('八戒你是个大刺猬')
    print('-------')
    print(text)
