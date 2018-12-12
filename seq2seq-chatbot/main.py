import time
import copy
import pdb
import nltk

import click
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2

sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

"""
Training model [optional args]
"""
@click.command()
@click.option('-dc', '--data-corpus', default='qajokes', help='Data corpus to use for training and inference',)
@click.option('-bs', '--batch-size', default=32, help='Batch size for training on minibatches',)
@click.option('-n', '--num-epochs', default=50, help='Number of epochs for training',)
@click.option('-lr', '--learning-rate', default=0.001, help='Learning rate to use when training model',)
@click.option('-inf', '--inference-mode', is_flag=True, help='Flag for INFERENCE mode',)
@click.option('-deb', '--debug', is_flag=True, help='Flag for setting a pdb trace after the setup')
@click.option('-lf', '--loss-filename', default='losses.txt', help='training/validation loss output filename')
@click.option('-t', '--test-loss', is_flag=True, help="Flag for evaluating a model's test loss")
@click.option('-ptr', '--pre-trained', is_flag=False, help="Flag for running a pretrained word embedding")
@click.option('-bl', '--baseline', is_flag=True, help='Flag for running a baseline test')
@click.option('-char', '--char-bool', is_flag=False, help='Flag for running a character model')
@click.option('-o', '--oracle', is_flag=True, help='Flag for running an oracle test')
@click.option('-rev', '--reverse', is_flag=False, help='Flag for reversing input sequence')
@click.option('-blu', '--bleutest', is_flag=False, help='Calculate the bleu value for the validation-set')
def train(data_corpus, batch_size, num_epochs, learning_rate, inference_mode, debug,
          loss_filename, test_loss, pre_trained, baseline, char_bool, oracle, reverse, bleutest):

    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup(data_corpus)

    if data_corpus == 'qajokes/glove': #pretrained model using gloVe
        data_corpus = 'qajokesGlove'
        pretrainedModelName = 'glove.6B.100d.txt'

    if data_corpus == 'qajokes/w2vec': #our own pretrained model
        data_corpus = 'qajokesW2Vec'
        pretrainedModelName = 'metadata.txt'

    if data_corpus == 'qajokes/char_all': #corpus with alphanumeric characters including punctuation
        data_corpus = 'qajokesCharAll'

    if data_corpus == 'qajokes/char': #corpus with only alphanum eric characters 
        data_corpus = 'qajokesChar'
    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    n_step = src_len // batch_size
    src_vocab_size = len(metadata['idx2w']) # 8002 (0~8001)
    emb_dim = 1024
    hidden_size = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word

    unk_id = word2idx['unk']   # 1
    pad_id = word2idx['_']     # 0

    start_id = src_vocab_size  # 8002
    end_id = src_vocab_size + 1  # 8003

    word2idx.update({'start_id': start_id})
    word2idx.update({'end_id': end_id})
    idx2word = idx2word + ['start_id', 'end_id']

    src_vocab_size = tgt_vocab_size = src_vocab_size + 2

    """ A data for Seq2Seq should look like this:
    input_seqs : ['how', 'are', 'you', '<PAD_ID'>]
    decode_seqs : ['<START_ID>', 'I', 'am', 'fine', '<PAD_ID'>]
    target_seqs : ['I', 'am', 'fine', '<END_ID>', '<PAD_ID'>]
    target_mask : [1, 1, 1, 1, 0]
    """
    # Preprocessing
    target_seqs = tl.prepro.sequences_add_end_id([trainY[10]], end_id=end_id)[0]
    decode_seqs = tl.prepro.sequences_add_start_id([trainY[10]], start_id=start_id, remove_last=False)[0]
    target_mask = tl.prepro.sequences_get_mask([target_seqs])[0]
    if not inference_mode:
        print("encode_seqs", [idx2word[id] for id in trainX[10]])
        print("target_seqs", [idx2word[id] for id in target_seqs])
        print("decode_seqs", [idx2word[id] for id in decode_seqs])
        print("target_mask", target_mask)
        print(len(target_seqs), len(decode_seqs), len(target_mask))

    # Init Session
    tf.reset_default_graph()
    sess = tf.Session(config=sess_config)
        
    # Training Data Placeholders
    encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
    decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
    target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
    target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") 

    if pre_trained:
        net_out, _ = create_model_pretrained(encode_seqs, decode_seqs, src_vocab_size, emb_dim,
                                hidden_size, pretrainedModelName, is_train=True, reuse=False)
    else:
        net_out, _ = create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim,
                                hidden_size, is_train=True, reuse=False)


    net_out.print_params(False)

    # Inference Data Placeholders
    encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
    decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")


    if pre_trained:
        net, net_rnn = create_model_pretrained(encode_seqs2, decode_seqs2, src_vocab_size, emb_dim,
                                    hidden_size, pretrainedModelName, is_train=False, reuse=True)
    else:
        net, net_rnn = create_model(encode_seqs2, decode_seqs2, src_vocab_size, emb_dim,
                                    hidden_size, is_train=False, reuse=True)

    y = tf.nn.softmax(net.outputs)

        

    # Loss Function
    loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, 
                                                input_mask=target_mask, return_details=False, name='cost')

    # Optimizer
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # Init Vars
    sess.run(tf.global_variables_initializer())


    # Load Model
    tl.files.load_and_assign_npz(sess=sess, name=data_corpus+'.model.npz', network=net)
    print("Loading " + data_corpus + " trained network")

    """
    Inference using pre-trained model
    """
    def inference(seed, char_bool = False):
        if char_bool == True:
            seed_id = [word2idx.get(w, unk_id) for w in list(seed)]
        else:
            seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")]

        if reverse: seed_id.reverse()
        
        # Encode and get state
        state = sess.run(net_rnn.final_state_encode,
                        {encode_seqs2: [seed_id]})
        # Decode, feed start_id and get first word [https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py]
        o, state = sess.run([y, net_rnn.final_state_decode],
                        {net_rnn.initial_state_decode: state,
                        decode_seqs2: [[start_id]]})
        o = o[0][2:]   # cut out '_' and 'unk'
        w_id = tl.nlp.sample_top(o, top_k=1) + 2 # add 2 to account for removed '_' and 'unk'
        # change top_k = 3 for more variation in output responses
        w = idx2word[w_id]
        # Decode and feed state iteratively
        sentence = [w]
        for _ in range(30): # max sentence length
            o, state = sess.run([y, net_rnn.final_state_decode],
                            {net_rnn.initial_state_decode: state,
                            decode_seqs2: [[w_id]]})
            o = o[0][2:]   # cut out '_' and 'unk'
            w_id = tl.nlp.sample_top(o, top_k=1) + 2 # add 2 to account for removed '_' and 'unk'
            w = idx2word[w_id]
            if w_id == end_id:
                break
            sentence = sentence + [w]
        return sentence

    def calculateBlEU(answers, questions, idx2word, char_bool):
        answer = []
        generatedResponse = []
        for i in tqdm(range(len(questions))):
            question = [idx2word[id] for id in questions[i]]
            answer.append([[idx2word[id] for id in answers[i]]])
            generatedResponse.append(inference(' '.join(question), char_bool))
            #BLEUscore += nltk.translate.bleu_score.sentence_bleu([answer], generatedResponse)
            #potentially include  smoothing_function=cc.method4
        return nltk.translate.bleu_score.corpus_bleu(answer, generatedResponse) #BLEUscore/len(questions)

    if inference_mode:
        print('Inference Mode')
        print('--------------')
        while True:
            input_seq = input('Enter Query: ')
            sentence = inference(input_seq, char_bool)
            print(" >", ' '.join(sentence))

    elif bleutest:
        BLEUscore = calculateBlEU(validX, validY, idx2word, char_bool)
        print('BLEU score {:.4f}'.format(BLEUscore))

    elif test_loss:  # evaluate and print the test loss
        testX, testY = shuffle(testX, testY, random_state=0)
        total_loss, n_iter = 0, 0
        for X, Y in tqdm(tl.iterate.minibatches(inputs=testX, targets=testY, batch_size=batch_size, shuffle=False), 
                        total=n_step, desc='test set loss computation', leave=False):
            
            if reverse: X = [x[::-1] for x in X]
            X = tl.prepro.pad_sequences(X)
            _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
            _target_seqs = tl.prepro.pad_sequences(_target_seqs)
            _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
            _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
            _target_mask = tl.prepro.sequences_get_mask(_target_seqs)
            loss_iter = sess.run(loss, {encode_seqs: X, decode_seqs: _decode_seqs,
                            target_seqs: _target_seqs, target_mask: _target_mask})
            total_loss += loss_iter
            n_iter += 1

        # printing test loss
        print('test loss {:.4f}'.format(total_loss / n_iter))

    else:
        seeds = ["happy birthday have a nice day",
                 "donald trump won last nights presidential debate according to snap online polls"]
        with open(loss_filename, 'w') as loss_file:
            loss_file.write('Training_Loss    Validation_Loss\n')
        for epoch in range(num_epochs):
            # compute training set loss and do training
            trainX, trainY = shuffle(trainX, trainY, random_state=0)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                             total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):
                
                if reverse: X = [x[::-1] for x in X]
                X = tl.prepro.pad_sequences(X)
                _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
                _target_seqs = tl.prepro.pad_sequences(_target_seqs)
                _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
                _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
                _target_mask = tl.prepro.sequences_get_mask(_target_seqs)
                if debug:
                    for i in range(len(X)):
                        print(i, [idx2word[id] for id in X[i]])
                        print(i, [idx2word[id] for id in Y[i]])
                        print(i, [idx2word[id] for id in _target_seqs[i]])
                        print(i, [idx2word[id] for id in _decode_seqs[i]])
                        print(i, _target_mask[i])
                        print(len(_target_seqs[i]), len(_decode_seqs[i]), len(_target_mask[i]))
                _, loss_iter = sess.run([train_op, loss], {encode_seqs: X, decode_seqs: _decode_seqs,
                                target_seqs: _target_seqs, target_mask: _target_mask})
                if debug: pdb.set_trace()
                total_loss += loss_iter
                n_iter += 1

            # printing average training loss after every epoch
            print('Epoch [{}/{}]: training loss   {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
            #print('calculateBlEU {:.4f}'.format(calculateBlEU(trainX, trainY, idx2word, char_bool)))
            with open(loss_filename, 'a') as loss_file: loss_file.write('{:.4f}'.format(total_loss / n_iter))

            # compute validation-set loss
            validX, validY = shuffle(validX, validY, random_state=0)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=validX, targets=validY, batch_size=batch_size, shuffle=False), 
                            total=n_step, desc='  validation set loss computation', leave=False):

                if reverse: X = [x[::-1] for x in X]                
                X = tl.prepro.pad_sequences(X)
                _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
                _target_seqs = tl.prepro.pad_sequences(_target_seqs)
                _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
                _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
                _target_mask = tl.prepro.sequences_get_mask(_target_seqs)
                loss_iter = sess.run(loss, {encode_seqs: X, decode_seqs: _decode_seqs,
                                target_seqs: _target_seqs, target_mask: _target_mask})
                total_loss += loss_iter
                n_iter += 1

            # printing average validation loss after every epoch
            print('              ' + ' '*(epoch >= 9) +'validation loss {:.4f}'.format(total_loss / n_iter))
            with open(loss_filename, 'a') as loss_file: loss_file.write(' '*11 + '{:.4f}\n'.format(total_loss / n_iter))
            
                
            # inference after every epoch
            for seed in seeds:
                print("Query >", seed)
                for _ in range(5):
                    sentence = inference(seed, char_bool)
                    print(" >", ' '.join(sentence))
            
            # saving the model
            tl.files.save_npz(net.all_params, name=data_corpus+'.model.npz', sess=sess)
    
    # session cleanup
    sess.close()



"""
Creates the LSTM Model
"""
def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, hidden_size, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = src_vocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = src_vocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.nn.rnn_cell.LSTMCell,
                n_hidden = hidden_size,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')

        net_out = DenseLayer(net_rnn, n_units=src_vocab_size, act=tf.identity, name='output')
    return net_out, net_rnn

"""
Creates the LSTM Model for pretrained network
"""
def create_model_pretrained(encode_seqs, decode_seqs, src_vocab_size, emb_dim, hidden_size, pretrainedModelName, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        # for chatbot, you can use the same embedding layer,
        # for translation, you may want to use 2 seperated embedding layers
        word2idx, weights, vocab_size, embedding_dim = loadGloveModel(pretrainedModelName)#'glove.6B.100d.txt')

        with tf.variable_scope("embedding") as vs: 
            glove_weights_initializer = tf.constant_initializer(weights)
            embedding_encode = EmbeddingInputlayer(inputs = encode_seqs,
                vocabulary_size = vocab_size,
                embedding_size = embedding_dim,
                E_init = glove_weights_initializer,
                name = 'seq_embedding')
            vs.reuse_variables()
            embedding_decode = EmbeddingInputlayer(inputs = decode_seqs,
                vocabulary_size = vocab_size,
                embedding_size = embedding_dim,
                E_init = glove_weights_initializer,
                name = 'seq_embedding')
            
            
        net_rnn = Seq2Seq(embedding_encode, embedding_decode,
                cell_fn = tf.nn.rnn_cell.LSTMCell,
                n_hidden = hidden_size,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')

        net_out = DenseLayer(net_rnn, n_units=src_vocab_size, act=tf.identity, name='output')
    return net_out, net_rnn


"""
Loads in pretrained network
"""
def loadGloveModel(gloveFile):
    print("Loading Pretrained Model")
    f = open(gloveFile, encoding="utf8")
    vocab_size_lim = 50000
    word2idx = {}
    i = 0
    weights = {}
    for line in f:
        if i <= vocab_size_lim:
            splitLine = line.split()
            word = splitLine[0]
            weights[word] = [np.float32(val) for val in splitLine[1:]]
            word2idx[word] = i
            i += 1
        else: 
            break

    vocab_size = len(weights)
    emb_dim = len(weights[list(weights.keys())[0]])
    W = np.zeros((vocab_size, emb_dim))
    i = 0
    for word in weights.keys():
        W[i, :] = weights[word]
        i += 1

    return word2idx, W, vocab_size, emb_dim

"""
Initial Setup
"""
def initial_setup(data_corpus):
    # import the data corpus (questions, answers, and metadata such as the vocab dict)
    import_str = 'from data.' + data_corpus.split('/')[0] + ' import data' 
    exec(import_str, globals())
    metadata, idx_q, idx_a = data.load_data(PATH='data/{}/'.format(data_corpus))

    # first 70% of dataset is training, second 15% is test, last 15% is validation
    (trainX, trainY), (testX, testY), (validX, validY) = data.split_dataset(idx_q, idx_a)
    trainX = remove_pad_sequences(trainX.tolist())
    trainY = remove_pad_sequences(trainY.tolist())
    testX = remove_pad_sequences(testX.tolist())
    testY = remove_pad_sequences(testY.tolist())
    validX = remove_pad_sequences(validX.tolist())
    validY = remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY


def remove_pad_sequences(sequences, pad_id=0):
    """Remove padding.

    Parameters
    -----------
    sequences : list of list of int
        All sequences where each row is a sequence.
    pad_id : int
        The pad ID.

    Returns
    ----------
    list of list of int
        The processed sequences.

    Examples
    ----------
    >>> sequences = [[2,3,4,0,0], [5,1,2,3,4,0,0,0], [4,5,0,2,4,0,0,0]]
    >>> print(remove_pad_sequences(sequences, pad_id=0))
    [[2, 3, 4], [5, 1, 2, 3, 4], [4, 5, 0, 2, 4]]

    """
    sequences_out = copy.deepcopy(sequences)

    for i, _ in enumerate(sequences):
        # for j in range(len(sequences[i])):
        #     if sequences[i][j] == pad_id:
        #         sequences_out[i] = sequences_out[i][:j]
        #         break
        for j in range(1, len(sequences[i])+1):
            if sequences[i][-j] != pad_id:
                sequences_out[i] = sequences_out[i][0:-j + 1]
                break

    return sequences_out

def main():
    try:
        train()
    except KeyboardInterrupt:
        print('Aborted!')

if __name__ == '__main__':
    main()
