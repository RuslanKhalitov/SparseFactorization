import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
import tensorflow as tf  # we use both tensorflow and pytorch (pytorch for main part) , tensorflow for tokenizer
from SMF_base import *
import matplotlib.pyplot as plt

from SMF_NLP_competitor import *

torch.manual_seed(42);
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 100


def read_IMDB():
    df = pd.read_csv('../input/IMDB Dataset.csv')
    # Convert sentiment columns to numerical values
    df.sentiment = df.sentiment.apply(lambda x: 1 if x=='positive' else 0)
    ## Cross validation
    # create new column "kfold" and assign a random value
    df['kfold'] = -1
    # Random the rows of data
    df = df.sample(frac=1).reset_index(drop=True)
    # get label
    y = df.sentiment.values
    # initialize kfold
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new values to kfold column
    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, 'kfold'] = fold
    df.head(3)
    return df


def read_Sentiment():
    df = pd.read_csv('../input/Sentiment140.csv')
    df.columns = ["sentiment", "id", "date", "flag", "user", "review"]
    # Convert sentiment columns to numerical values
    df.sentiment = df.sentiment.apply(lambda x: 1 if x=='4' else 0)
    ## Cross validation
    # create new column "kfold" and assign a random value
    df['kfold'] = -1
    # Random the rows of data
    df = df.sample(frac=1).reset_index(drop=True)
    # get label
    y = df.sentiment.values
    # initialize kfold
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new values to kfold column
    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, 'kfold'] = fold
    df.head(3)
    return df


def read_corona():
    df_train = pd.read_csv('../input/Corona_NLP_train.csv')
    df_test = pd.read_csv('../input/Corona_NLP_test.csv')
    df = pd.concat([df_train, df_test])
    df.columns = ["UserName", "ScreenName", "Location", "TweetAt", "review", "sentiment"]
    # Convert sentiment columns to numerical values
    df.sentiment = df.sentiment.apply(lambda x: 5 if x == 'Extremely Positive' else 4 if x == "Positive"
    else 3 if x=="Neural" else 2 if x == "Negative" else 1)
    ## Cross validation
    # create new column "kfold" and assign a random value
    df['kfold'] = -1
    # Random the rows of data
    df = df.sample(frac=1).reset_index(drop=True)
    # get label
    y = df.sentiment.values
    # initialize kfold
    kf = model_selection.StratifiedKFold(n_splits=5)
    # fill the new values to kfold column
    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, 'kfold'] = fold
    df.head(3)
    return df


def load_glove():
    glove = pd.read_csv('../input/glove.6B.100d.txt', sep=" ", quoting=3,
                        header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    # Check check the dimension of this fasttext embedding version
    print(glove_embedding['hello'].shape)
    return glove_embedding


class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array

        Return xtrain and ylabel in torch tensor datatype, stored in dictionary format
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        # return length of dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # given an idex (item), return review and target of that index in torch tensor
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = torch.tensor(self.target[index], dtype=torch.float)

        return {'review': review,
                'target': target}


class SentimentDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array

        Return xtrain and ylabel in torch tensor datatype, stored in dictionary format
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        # return length of dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # given an idex (item), return review and target of that index in torch tensor
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = torch.tensor(self.target[index], dtype=torch.float)

        return {'review': review,
                'target': target}


class CoronaDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array

        Return xtrain and ylabel in torch tensor datatype, stored in dictionary format
        """
        self.reviews = reviews
        self.target = targets

    def __len__(self):
        # return length of dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # given an idex (item), return review and target of that index in torch tensor
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = torch.tensor(self.target[index], dtype=torch.float)

        return {'review': review,
                'target': target}
      
      
def create_embedding_matrix(word_index, embedding_dict=None, d_model=100):
    """
     this function create the embedding matrix save in numpy array
    :param word_index: a dictionary with word: index_value
    :param embedding_dict: a dict with word embedding
    :d_model: the dimension of word pretrained embedding, here I just set to 100, we will define again
    :return a numpy array with embedding vectors for all known words
    """
    embedding_matrix = np.zeros((len(word_index) + 1, d_model))
    ## loop over all the words
    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix


def train(data_loader, model, optimizer, device):
    """
    this is model training for one epoch
    data_loader:  this is torch dataloader, just like dataset but in torch and devide into batches
    model : lstm
    optimizer : torch optimizer : adam
    device:  cuda or cpu
    """
    # set model to training mode
    model.train()
    # go through batches of data in data loader
    train_loss = 0.0
    final_predictions = []
    final_targets = []
    criterion = nn.BCEWithLogitsLoss()
    for data in data_loader:
        reviews = data['review']
        targets = data['target']
        # move the data to device that we want to use
        reviews = reviews.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # clear the gradient
        optimizer.zero_grad()
        # make prediction from model
        predictions = model(reviews)
        # caculate the losses
        loss = criterion(predictions, targets.view(-1,1))
        # backprob
        loss.backward()
        train_loss += loss
        #single optimization step
        optimizer.step()
        predictions = predictions.cpu().detach().numpy().tolist()
        targets = data['target'].cpu().detach().numpy().tolist()
        final_predictions.extend(predictions)
        final_targets.extend(targets)

    return final_predictions, final_targets, train_loss


def evaluate(data_loader, model, device):
    final_predictions = []
    final_targets = []
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    # turn off gradient calculation
    with torch.no_grad():
        val_loss = 0.0
        for data in data_loader:
            reviews = data['review']
            targets = data['target']
            reviews = reviews.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            # make prediction
            predictions = model(reviews)
            # move prediction and target to cpu
            loss = criterion(predictions, targets.view(-1,1))
            # add predictions to final_prediction
            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()
            val_loss += loss
            final_predictions.extend(predictions)
            final_targets.extend(targets)
    return final_predictions, final_targets, val_loss


def training():
    df = read_IMDB()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())
    print('Load Glove embedding')
    glove_embedding = load_glove()
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, d_model=100)
    print(embedding_matrix.shape)
    cfg = {
        'n_class': [1],
        'n_hidden_f': [20],
        'n_hidden_g': [3],
        'N': [MAX_LEN],
        'd': [100],
        'n_W': [7],
        'with_g': [True],
        'masking': [True],
        'residual': [True]
    }
    for fold in range(1):
        # STEP 2: cross validation
        train_df = df[df.kfold != fold].reset_index(drop=True)
        valid_df = df[df.kfold == fold].reset_index(drop=True)

        # STEP 3: pad sequence
        xtrain = tokenizer.texts_to_sequences(train_df.review.values)
        xtest = tokenizer.texts_to_sequences(valid_df.review.values)

        # zero padding
        xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
        xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

        # STEP 4: initialize dataset class for training
        train_dataset = IMDBDataset(reviews=xtrain, targets=train_df.sentiment.values)

        # STEP 5: Load dataset to Pytorch DataLoader
        # after we have train_dataset, we create a torch dataloader to load train_dataset class based on specified batch_size
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=2)
        # initialize dataset class for validation
        valid_dataset = IMDBDataset(reviews=xtest, targets=valid_df.sentiment.values)
        valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=1)

        # STEP 6: Running
        device = torch.device('cuda')
        # feed embedding matrix to lstm
        model_glove = InteractionModuleEmbedSkip(
            embedding_matrix,
            cfg["n_class"][0],
            cfg["n_W"][0],
            cfg["N"][0],
            cfg["d"][0],
            masking=cfg['masking'][0],
            with_g=cfg['with_g'][0],
            residual_every=cfg['residual'][0]
        )
        #model_glove = LSTM(embedding_matrix)
        print(model_glove)
        # set model to cuda device
        model_glove.to(device)
        # initialize Adam optimizer
        optimizer = torch.optim.Adam([param for param in model_glove.parameters() if param.requires_grad==True], lr=1e-5)

        print('training model')
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        for epoch in range(EPOCHS):
            # train one epoch
            train_predictions, train_targets, train_loss = train(train_data_loader, model_glove, optimizer, device)
            train_losses.append(train_loss.cpu().detach().numpy())
            # validate
            outputs, targets, val_loss = evaluate(valid_data_loader, model_glove, device)
            val_losses.append(val_loss.cpu().detach().numpy())
            # threshold
            train_outputs = np.array(train_predictions) >= 0.5
            val_outputs = np.array(outputs) >= 0.5
            # calculate accuracy
            train_accuracy = metrics.accuracy_score(train_targets, train_outputs)
            val_accuracy = metrics.accuracy_score(targets, val_outputs)
            print(f'FOLD:{fold}, epoch: {epoch}, val_accuracy_score: {val_accuracy}, val_loss:{val_loss}')
            val_accs.append(val_accuracy)
            train_accs.append(train_accuracy)

        plt.gca().cla()
        plt.subplot(2, 2, 1)
        plt.plot(train_accs, '*')
        plt.xlabel('n_epochs')
        plt.ylabel('train_acc')
        plt.subplot(2, 2, 2)
        plt.plot(train_losses)
        plt.xlabel('n_epochs')
        plt.ylabel('train_loss')
        plt.yscale('log')
        plt.subplot(2, 2, 3)
        plt.plot(val_accs, '*')
        plt.xlabel('n_epochs')
        plt.ylabel('val_acc')
        plt.subplot(2, 2, 4)
        plt.plot(val_losses)
        plt.xlabel('n_epochs')
        plt.ylabel('val_loss')
        plt.yscale('log')
        plt.savefig('smf100_gtrue.png')


if __name__ == '__main__':
    training()
    #print(read_Sentiment())
