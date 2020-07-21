#字符级模型
import logging
import time
from enum import Enum
import heapq
import numpy as np
import tensorflow as tf

logging.getLogger('tensorflow').setLevel(logging.WARNING)
SampleType = Enum('SampleType',('weighted_sample','select_given'))

# CharRNNLM(is_training=False,w2v_model = self.w2v.model,vocab_size=w2v_vocab_size, infer=True, **params)
class CharRNNLM(object):
    def __init__(self, is_training, batch_size, num_unrollings, vocab_size,w2v_model,
                 hidden_size, max_grad_norm, embedding_size, num_layers,
                 learning_rate, cell_type, dropout=0.0, input_dropout=0.0, infer=False):
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        if infer:
            self.batch_size = 1
            self.num_unrollings = 1
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.w2v_model = w2v_model

        if embedding_size <= 0:
            self.input_size = vocab_size
            self.input_dropout = 0.0
        else:
            self.input_size = embedding_size

        self.input_data = tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='inputs')
        self.targets =  tf.placeholder(tf.int64, [self.batch_size, self.num_unrollings], name='targets')

        #定义单步cell
        cell_fn = tf.nn.rnn_cell.BasicLSTMCell

        params = dict()
        #params = {'input_size': self.input_size}
        params['forget_bias'] = 1.0  # 1.0 is default value
        cell = cell_fn(self.hidden_size, **params)

        cells = [cell]
        #params['input_size'] = self.hidden_size
        for i in range(self.num_layers-1):
            higher_layer_cell = cell_fn(self.hidden_size, **params)
            cells.append(higher_layer_cell)

        if is_training and self.dropout > 0:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0-self.dropout) for cell in cells]#输出部分作为下一层的输入

        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        with tf.name_scope('initial_state'):
            self.initial_state = tuple(
                        [tf.nn.rnn_cell.LSTMStateTuple(#存储LSTM单元的state_size,zero_state和output state的元组
                            tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size[idx][0]],
                                          'initial_lstm_state_'+str(idx+1)),
                            tf.placeholder(tf.float32, [self.batch_size, multi_cell.state_size[idx][1]],
                                           'initial_lstm_state_'+str(idx+1)))
                            for idx in range(self.num_layers)])

        with tf.name_scope('embedding_layer'):
            if embedding_size > 0:
                self.embedding = tf.get_variable("word_embeddings",
                    initializer=self.w2v_model.vectors.astype(np.float32))
            else:
                self.embedding = tf.constant(np.eye(self.vocab_size), dtype=tf.float32)
            # 主要是选取一个张量里面索引对应的元素
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_data)
            if is_training and self.input_dropout > 0:
                inputs = tf.nn.dropout(inputs, 1-self.input_dropout)

        with tf.name_scope('slice_inputs'):
            # num_unrollings * (batch_size, embedding_size), the format of rnn inputs.
            sliced_inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(
                axis = 1, num_or_size_splits = self.num_unrollings, value = inputs)]

        # sliced_inputs: list of shape xx
        # inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size]
        # initial_state: An initial state for the RNN.
        #                If cell.state_size is an integer, this must be a Tensor of appropriate
        #                type and shape [batch_size, cell.state_size]
        # outputs: a length T list of outputs (one for each input), or a nested tuple of such elements.
        # state: the final state
        #使用指定的RNN神经元创建循环神经网络
        outputs, final_state = tf.nn.static_rnn(
                cell = multi_cell,
                inputs = sliced_inputs,
                initial_state=self.initial_state)
        self.final_state = final_state

        with tf.name_scope('flatten_outputs'):
            flat_outputs = tf.reshape(tf.concat(axis = 1, values = outputs), [-1, hidden_size])

        with tf.name_scope('flatten_targets'):
            flat_targets = tf.reshape(tf.concat(axis = 1, values = self.targets), [-1])

        with tf.variable_scope('softmax') as sm_vs:
            softmax_w = tf.get_variable('softmax_w', [hidden_size, vocab_size])
            softmax_b = tf.get_variable('softmax_b', [vocab_size])
            self.logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
            self.probs = tf.nn.softmax(self.logits)

        with tf.name_scope('loss'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = self.logits, labels = flat_targets)
            self.mean_loss = tf.reduce_mean(loss)

        with tf.name_scope('loss_montor'):
            count = tf.Variable(1.0, name='count')
            sum_mean_loss = tf.Variable(1.0, name='sum_mean_loss')

            self.reset_loss_monitor = tf.group(sum_mean_loss.assign(0.0),
                                               count.assign(0.0), name='reset_loss_monitor')
            self.update_loss_monitor = tf.group(sum_mean_loss.assign(sum_mean_loss+self.mean_loss),
                                                count.assign(count+1), name='update_loss_monitor')

            with tf.control_dependencies([self.update_loss_monitor]):
                self.average_loss = sum_mean_loss / count
                self.ppl = tf.exp(self.average_loss)

            average_loss_summary = tf.summary.scalar(
                    name = 'average loss', tensor = self.average_loss)
            ppl_summary = tf.summary.scalar(
                    name = 'perplexity', tensor = self.ppl)

        self.summaries = tf.summary.merge(
                inputs = [average_loss_summary, ppl_summary], name='loss_monitor')

        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0.0))

        # self.learning_rate = tf.constant(learning_rate)
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        if is_training:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)


    def run_epoch(self, session, batch_generator, is_training, learning_rate, verbose=0, freq=10):
        epoch_size = batch_generator.num_batches

        if verbose > 0:
            logging.info('epoch_size: %d', epoch_size)
            logging.info('data_size: %d', batch_generator.seq_length)
            logging.info('num_unrollings: %d', self.num_unrollings)
            logging.info('batch_size: %d', self.batch_size)

        if is_training:
            extra_op = self.train_op
        else:
            extra_op = tf.no_op()


        state = tuple([(np.zeros((self.batch_size, self.hidden_size)),
                np.zeros((self.batch_size, self.hidden_size)))
                for _ in range(self.num_layers)])

        self.reset_loss_monitor.run()
        batch_generator.reset_batch_pointer()
        start_time = time.time()
        ppl_cumsum = 0
        for step in range(epoch_size):
            x, y = batch_generator.next_batch()

            ops = [self.average_loss, self.ppl, self.final_state, extra_op,
                   self.summaries, self.global_step]

            feed_dict = {self.input_data: x, self.targets: y, self.initial_state: state,
                         self.learning_rate: learning_rate}

            results = session.run(ops, feed_dict)
            average_loss, ppl, final_state, _, summary_str, global_step = results
            ppl_cumsum += ppl

            # if (verbose > 0) and ((step+1) % freq == 0):
            if ((step+1) % freq == 0):
                logging.info('%.1f%%, step:%d, perplexity: %.3f, speed: %.0f words',
                             (step + 1) * 1.0 / epoch_size * 100, step, ppl_cumsum/(step+1),
                             (step + 1) * self.batch_size * self.num_unrollings / (time.time() - start_time))
        logging.info("Perplexity: %.3f, speed: %.0f words per sec",
                     ppl, (step + 1) * self.batch_size * self.num_unrollings / (time.time() - start_time))

        return ppl, summary_str, global_step

    def sample_seq(self, session, length, start_text):
        state = tuple([(np.zeros((self.batch_size, self.hidden_size)),
                np.zeros((self.batch_size, self.hidden_size)))
                for _ in range(self.num_layers)])

        # use start_text to warm up the RNN.
        start_text = self.check_start(start_text)
        if start_text is not None and len(start_text) > 0:
            seq = list(start_text)
            for char in start_text[:-1]:
                x = np.array([[self.w2v_model.vocab_hash[char]]])
                # state = session.run(self.final_state, {self.input_data: x, self.initial_state: state})
            x = np.array([[self.w2v_model.vocab_hash[start_text[-1]]]])
        else:
            x = np.array([[np.random.randint(0, self.vocab_size)]])
            seq = []

        for i in range(length):
        	#激活值和预测值
            state, logits = session.run([self.final_state, self.logits],
                                        {self.input_data: x, self.initial_state: state})
            unnormalized_probs = np.exp(logits[0] - np.max(logits[0]))
            probs = unnormalized_probs / np.sum(unnormalized_probs)

            sample = np.random.choice(self.vocab_size, 1, p=probs)[0] #随机采样

            seq.append(self.w2v_model.vocab[sample])
            x = np.array([[sample]])

        return ''.join(seq)

    #确定输入的文字在字典中
    def check_start(self,text):
        idx = text.find('<')
        if idx > -1:
            text = text[:idx]

        valid_text = []
        for w in text:
            if w in self.w2v_model.vocab:
                valid_text.append(w)
        return ''.join(valid_text)

