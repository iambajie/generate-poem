#设置相关参数
import argparse
import numpy as np
import time

#训练参数
def config_poem_train(args=''):
    parser = argparse.ArgumentParser()

    #数据路径
    parser.add_argument('--data_path', type=str,
                        default='./data/poem/',
                        help='data path')


    parser.add_argument('--encoding', type=str,
                        default='utf-8',
                        help='the encoding of the data file.')

    #保存输出模型
    parser.add_argument('--output_dir', type=str, default='output_model',
                        help=('directory to store final and'
                              ' intermediate results and models.'))
    #保存最佳模型
    parser.add_argument('--init_dir', type=str, default='',
                        help='continue from the outputs in the given directory')

    #网络设置
    parser.add_argument('--hidden_size', type=int, default=128,#128,
                        help='size of RNN hidden state vector')
    parser.add_argument('--embedding_size', type=int, default=128,#0,
                        help='size of character embeddings, 0 for one-hot')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--num_unrollings', type=int, default=64,#10,
                        help='number of unrolling steps.')
    parser.add_argument('--cell_type', type=str, default='lstm',
                        help='which model to use (rnn, lstm or gru).')

    #训练参数设置
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='minibatch size')
    parser.add_argument('--train_frac', type=float, default=0.9,
                        help='fraction of data used for training.')
    parser.add_argument('--valid_frac', type=float, default=0.05,
                        help='fraction of data used for validation.')
    #测试集大小1-0.9-0.05=0.05
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate, default to 0 (no dropout).')

    parser.add_argument('--input_dropout', type=float, default=0.0,
                        help=('dropout rate on input layer, default to 0 (no dropout),'
                              'and no dropout if using one-hot representation.'))

    #梯度下降
    #修剪梯度设置的最大值
    parser.add_argument('--max_grad_norm', type=float, default=5.,
                        help='clip global grad norm')
    parser.add_argument('--learning_rate', type=float, default=5e-3,
                        help='initial learning rate')

    #打印日志
    parser.add_argument('--progress_freq', type=int, default=100,
                        help=('frequency for progress report in training and evalution.'))
    parser.add_argument('--verbose', type=int, default=0,
                        help=('whether to show progress report in training and evalution.'))

    #初始模型和当前最佳模型的参数
    parser.add_argument('--init_model', type=str,
                        default='', help=('initial model'))
    parser.add_argument('--best_model', type=str,
                        default='', help=('current best model'))
    parser.add_argument('--best_valid_ppl', type=float,
                        default=np.Inf, help=('current valid perplexity'))

    # # Parameters for using saved best models.
    # parser.add_argument('--model_dir', type=str, default='',
    #                     help='continue from the outputs in the given directory')

    # parser.add_argument('--debug', dest='debug', action='store_true',
    #                     help='show debug information')
    # parser.set_defaults(debug=False)

    #测试
    parser.add_argument('--test', dest='test', action='store_true',
                        help=('use the first 1000 character to as data to test the implementation'))
    parser.set_defaults(test=False)

    #解析参数
    args = parser.parse_args(args.split())

    return args



def config_sample(args=''):  #('--model_dir output_poem --length 16 --seed {}'.format(now))
    parser = argparse.ArgumentParser()

    # hyper-parameters for using saved best models.
    # 学习日志和结果相关的超参数
    logging_args = parser.add_argument_group('Logging_Options')
    logging_args.add_argument('--model_dir', type=str,
            default='demo_model/',
            help='continue from the outputs in the given directory')

    logging_args.add_argument('--data_dir', type=str,
            default='./data/poem',
            help='data file path')

    logging_args.add_argument('--best_model', type=str,
            default='', help=('current best model'))

    # hyper-parameters for sampling.
    # 设置sampling相关的超参数
    testing_args = parser.add_argument_group('Sampling Options')
    testing_args.add_argument('--max_prob', dest='max_prob', action='store_true',
                        help='always pick the most probable next character in sampling')
    testing_args.set_defaults(max_prob=False)

    testing_args.add_argument('--start_text', type=str,
                        default='The meaning of life is ',
                        help='the text to start with')

    testing_args.add_argument('--length', type=int,
                        default=100,
                        help='length of sampled sequence')

    testing_args.add_argument('--seed', type=int,
                        default=-1,
                        help=('seed for sampling to replicate results, '
                              'an integer between 0 and 4294967295.'))

    args = parser.parse_args(args.split())

    return args
if __name__ == '__main__':
    now = int(time.time())
    args = config_sample('--model_dir output_poem --length 16 --seed {}'.format(now))
    print(args)