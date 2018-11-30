EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz ' # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''

FILENAME = 'qajokes.txt'
JOKE_FILES = ('jokes.csv', 'lightbulbs.csv', 'nosubject.csv', 'qajokes.csv')


UNK = 'unk'
VOCAB_SIZE = 44000

import random
import sys
import csv
import itertools
from collections import defaultdict
import pickle
import pdb

import click
import nltk
import numpy as np


def ddefault():
    return 1

'''
 read lines from file
     return [list of lines]

'''
def read_lines(filename):
    return open(filename).read().split('\n')[:-1]


'''
 split sentences in one line
  into multiple lines
    return [list of lines]

'''
def split_line(line):
    return line.split('.')


'''
 remove anything that isn't in the vocabulary
    return str(pure ta/en)

'''
def filter_line(line, whitelist):
    return ''.join([ ch for ch in line if ch in whitelist ])


'''
 read list of words, create index to word,
  word to index dictionaries
    return tuple( vocab->(word, count), idx2w, w2idx )

'''
def index_(tokenized_sentences, vocab_size):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = ['_'] + [UNK] + [ x[0] for x in vocab ]
    # word2index
    word2index = dict([(w,i) for i,w in enumerate(index2word)] )
    return index2word, word2index, freq_dist


'''
 filter too long and too short sequences
    return tuple( filtered_ta, filtered_en )

'''
def filter_data(sequences, limit):
    filtered_q, filtered_a = [], []
    raw_data_len = len(sequences)//2

    for i in range(0, len(sequences), 2):
        qlen, alen = len(sequences[i].split(' ')), len(sequences[i+1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                filtered_q.append(sequences[i])
                filtered_a.append(sequences[i+1])

    # print the fraction of the original data, filtered
    filt_data_len = len(filtered_q)
    filtered = int((raw_data_len - filt_data_len)*100/raw_data_len)
    print(str(filtered) + '% filtered from original data due to size limits')

    return filtered_q, filtered_a


'''
 create the final dataset :
  - convert list of items to arrays of indices
  - add zero padding
      return ( [array_en([indices]), array_ta([indices]) )

'''
def zero_pad(qtokenized, atokenized, w2idx, limit):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa']], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'])
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'])

        #print(len(idx_q[i]), len(q_indices))
        #print(len(idx_a[i]), len(a_indices))
        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)

    return idx_q, idx_a


'''
 replace words with indices in a sequence
  replace with unknown if word not in lookup
    return [list of indices]

'''
def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))


def process_data():

    limit = {
            'maxq' : 20,
            'minq' : 0,
            'maxa' : 20,
            'mina' : 1
            }

    print('\n>> Read lines from file')
    lines = read_lines(filename=FILENAME)

    # change to lower case (just for en)
    lines = [ line.lower() for line in lines ]

    print('\n:: Sample from read(p) lines')
    print(lines[121:125])

    # filter out unnecessary characters
    print('\n>> Filter lines')
    lines = [ filter_line(line, EN_WHITELIST) for line in lines ]
    print(lines[121:125])

    # filter out too long or too short sequences
    print('\n>> 2nd layer of filtering')
    qlines, alines = filter_data(lines, limit)
    print('\nq : {0} ; a : {1}'.format(qlines[60], alines[60]))
    print('\nq : {0} ; a : {1}'.format(qlines[61], alines[61]))


    # convert list of [lines of text] into list of [list of words ]
    print('\n>> Segment lines into words')
    qtokenized = [ list(filter(None, wordlist.split(' '))) for wordlist in qlines ]
    atokenized = [ list(filter(None, wordlist.split(' '))) for wordlist in alines ]
    index_and_save(qtokenized, atokenized, '.', VOCAB_SIZE, limit)

    print('\n>> Segment lines into characters')
    limit = {
            'maxq' : 200,
            'minq' : 0,
            'maxa' : 200,
            'mina' : 1
            }
    qtokenized = [ [x for x in sentence] for sentence in qlines ]
    atokenized = [ [x for x in sentence] for sentence in alines ]
    # TODO: also build a model where all characters are allowed (no whitelist)
    index_and_save(qtokenized, atokenized, 'char', len(EN_WHITELIST), limit)
    


def index_and_save(qtokenized, atokenized, path, vocab_size, limit):
    """ 
    Given lists of tokenized sequences of questions and answers, assembles vocab indices
    and indexed sequences and saves the data into objects that are ready for the training
    and inference scripts to use
    """
    print('\nq : {0} ; a : {1}'.format(qtokenized[60], atokenized[60]))
    print('\nq : {0} ; a : {1}'.format(qtokenized[61], atokenized[61]))


    # indexing -> idx2w, w2idx : en/ta
    print('\n >> Index words')
    idx2w, w2idx, freq_dist = index_( qtokenized + atokenized, vocab_size)

    print('\n >> Zero Padding')
    idx_q, idx_a = zero_pad(qtokenized, atokenized, w2idx, limit)

    print('\n >> Save numpy arrays to disk')
    # save them
    np.save(path+'/idx_q.npy', idx_q)
    np.save(path+'/idx_a.npy', idx_a)

    # let us now save the necessary dictionaries
    metadata = {
            'w2idx' : w2idx,
            'idx2w' : idx2w,
            'limit' : limit,
            'freq_dist' : freq_dist
                }

    # write to disk : data control dictionaries
    with open(path+'/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)


def compile_jokes():
    '''
    Assembles multiple joke csv files into a single text file of alternating
    query-response pairs. This format is consistent with the seq2seq chatbot code from the
    Tensorlayer tutorial and allows more code re-use
    '''
    # Display which files are being assembled
    print('compiling the following files into qajokes.txt:')
    s = '\t'
    for i, fn in enumerate(JOKE_FILES):
        s += fn
        if i < len(JOKE_FILES) - 1: s += ', ' 
    print(s)

    # Assemble all the csv's into a text file 
    with open(FILENAME, 'w') as outfile:
        for fn in JOKE_FILES:
            with open(fn, newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                for i, row in enumerate(csvreader):
                    if i == 0: continue
                    outfile.write('{}\n{}\n'.format(row[0].strip(), row[1].strip()))


def load_data(PATH=''):
    # read data control dictionaries
    try:
        with open(PATH + 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
    except:
        metadata = None
    # read numpy arrays
    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')
    return metadata, idx_q, idx_a

import numpy as np
from random import sample

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_dataset(x, y, ratio = [0.7, 0.15, 0.15] ):
    # number of examples
    data_len = len(x)
    lens = [ int(data_len*item) for item in ratio ]

    trainX, trainY = x[:lens[0]], y[:lens[0]]
    testX, testY = x[lens[0]:lens[0]+lens[1]], y[lens[0]:lens[0]+lens[1]]
    validX, validY = x[-lens[-1]:], y[-lens[-1]:]

    return (trainX,trainY), (testX,testY), (validX,validY)


'''
 generate batches from dataset
    yield (x_gen, y_gen)

    TODO : fix needed

'''
def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

#'''
# convert indices of alphabets into a string (word)
#    return str(word)
#
#'''
#def decode_word(alpha_seq, idx2alpha):
#    return ''.join([ idx2alpha[alpha] for alpha in alpha_seq if alpha ])
#
#
#'''
# convert indices of phonemes into list of phonemes (as string)
#    return str(phoneme_list)
#
#'''
#def decode_phonemes(pho_seq, idx2pho):
#    return ' '.join( [ idx2pho[pho] for pho in pho_seq if pho ])


'''
 a generic decode function
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])



if __name__ == '__main__':
    try:
        with open('qajokes.txt', 'r'):
            print('qajokes.txt found, using this file as training data')
            print('\tdelete this file if you want to generate a new training corpus')
    except: 
        print('no qajokes.txt found, compiling it now')
        compile_jokes()
    process_data()
