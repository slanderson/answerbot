
This directory contains code based on the seq2seq tutorial provided in the TensorLayer documentation.  The original
can be found [at their tutorial](https://github.com/tensorlayer/seq2seq-chatbot)

Th original code was modified and a new data corpus was added. The corpus is  150k question-answer
joke pairs. The code additions allow for pre-trained embeddings, training on reversed queries, character-based
training models. Additionally, the user can calculate the bleu corpus score of the resulting model.

# FunnyBot

This is a joke generater modeled after a seq2seq chatbot implementation  
- [Practical-Seq2Seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (optional)

### Prerequisites

- Python 
- [TensorFlow](https://github.com/tensorflow/tensorflow) >= 1.2
- [TensorLayer](https://github.com/zsdonghao/tensorlayer) >= 1.6.3
- download glove.6B.100d.txt (https://nlp.stanford.edu/projects/glove/)
- run word2vec_basic.py (a modified python tutorial)

### Model

<table class="image">
<div align="center">
    <img src="http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png"/>  
    <br>  
    <em align="center"></em>  
</div>
</table>


### Data Corpuses
qajokes "Corpus of word vectors of 150k question and answer joke pairs"
qajokes/char "Corpus of 150k character vectors without punctuation of 150k question and answer joke pairs"
qajokes/char_all "Corpus of 150k character vectors including punctuation of 150k question and answer joke pairs"

### Pretrained Embeddings and Data Corpuses
qajokes/w2vec "Corpus of word vectors of 150k question and answer joke pairs/ pretrained Word2Vector embedding trained 
                on the words in the corpus"
qajokes/glove "Corpus of word vectors of 150k question and answer joke pairs/ 50k most common words from pretrained 
                gloVe embedding"

### Training

```
python main.py --batch-size 32 --num-epochs 50 -lr 0.001
```

### Training with pretrained data
```
python main.py --batch-size 32 -dc qajokes/w2vec -ptr True --num-epochs 50 -lr 0.001 
```



### Calculate bleu score for validation set
```
python main.py -dc qajokes -blu True
```

### Calculate bleu score for validation set with pretrained data
```
python main.py -dc qajokes -blu True -blu True
```



### Inference
```
python main.py -dc qajokes -inf
```

### Inference with pretrained data

```
python main.py -dc qajokes/w2vec =ptr True -inf
```

### Results

Enter Query: why did the chicken cross the road
 > to get to the idiots house knock knock whos there the chicken

Enter Query: how many engineers does it take to change a lightbulb
 > none they dont change anything they just sit in the dark
 
Enter Query: how many lawyers does it take to chage a lightbulb
 > one but the lightbulb has to want to change
