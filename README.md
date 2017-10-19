# Sentence Pair Detector

  A siamese LSTM based approach to indentifying similar sentences/question. Built for the [quora question pair challenge](https://www.kaggle.com/c/quora-question-pairs) on Kaggle, but can be trained on any similar-sentence dataset. 
### Word Embeddings: 
The word embeddings that I used were the [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings. I used them mainly for their small size, but can be switched out for any other word embeddings. I would personally recommend using [Google's Word2Vec](https://code.google.com/archive/p/word2vec/) pretrained vectors.
### Preprocessing:
I cleaned the dataset to remove sentence contractions. I used the list of contractions given by [lystdo](https://www.kaggle.com/lystdo) on Kaggle. Standard preprocessing tasks like removing outliers in terms of size and stemming was done. The preprocessing functions are in the file [vectorize_input.py](https://github.com/rpanchal1996/question-pair/blob/master/vectorize_input.py).
### Model:
I used stacked LSTM cells with tied weights and dropout. The two sentences/questions were fed into the two parallel networks with equal weights, and the loss was calculated based on their output. The loss function is Yann LeCun's [Contrastive Loss Function](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf). The optimizer used is Adam. 

### To-Do:

 - Expand Readme for better explanation of the model
 - Remove hardcoded training parameters
 - Split and modularize the cleaning functions
 - Remove redundant one-time use functions
 - Add option for other optimizers and GRU cells
 - Add credits for resources used
 - ADD COMMENTS

 

 

