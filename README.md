# Similar sentence identification

  A siamese LSTM based approach to [quora question pair challenge](https://www.kaggle.com/c/quora-question-pairs) on Kaggle. 

The word embeddings that I used were the [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch) embeddings. I used them mainly for their small size. 
I have used two LSTMs with tied weights and a contrastive loss function to train the model. 
