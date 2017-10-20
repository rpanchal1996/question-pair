# Sentence Pair Detector

  A siamese LSTM based approach to indentifying similar sentences/question. Built for the [quora question pair challenge](https://www.kaggle.com/c/quora-question-pairs) on Kaggle, but can be trained on any similar-sentence dataset. 
  
### File Description:
- `generate_stemmed_vectors.py` processes the wordvectors. Stems the word associated with the vector and generates the final wordlist.
- `vectorize_input.py` contains a major portion of the preprocessing steps including cleaning of the data and generation of sentence vectors.
- `model.py` contains the actual model and batching function.
### How to train:
- Download the quora question pair dataset from [here](https://www.kaggle.com/c/quora-question-pairs) and extract the train.csv file
- Download the ConceptNet Numberbatch pretrained embedding from [here](https://github.com/commonsense/conceptnet-numberbatch)
- Run the vector processing script with the location of the embeddings of as an argument like: `python generate_stemmed_vectors.py "[path_to_numberbatch_embeddings]"`
- Run the rest of the preprocessing with the location of the quora `train.csv` as an argument like this: `python vectorize_input.py "[path to train dataset]"`
- Start the training by running `model.py`.
### An overview of what I have done:  
#### Word Embeddings: 
The word embeddings that I used were the [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings. I used them mainly for their small size, but can be switched out for any other word embeddings. I would personally recommend using [Google's Word2Vec](https://code.google.com/archive/p/word2vec/) pretrained vectors.
#### Preprocessing:
I cleaned the dataset to remove sentence contractions. I used the list of contractions given by [lystdo](https://www.kaggle.com/lystdo) on Kaggle. Standard preprocessing tasks like removing outliers in terms of size and stemming was done. The preprocessing functions are in the file [vectorize_input.py](https://github.com/rpanchal1996/question-pair/blob/master/vectorize_input.py).
#### Model:
I used stacked LSTM cells with tied weights and dropout. The two sentences/questions were fed into the two parallel networks with equal weights, and the loss was calculated based on their output. The loss function is Yann LeCun's [Contrastive Loss Function](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf). The optimizer used is Adam. 

### To-Do:

 - Expand Readme for better explanation of the model
 - Remove hardcoded training parameters
 - Add option for other optimizers and GRU cells
 - Add credits for resources used
 - ADD COMMENTS
