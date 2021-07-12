import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf  # we use both tensorflow and pytorch (pytorch for main part) , tensorflow for tokenizer
import matplotlib.pyplot as plt
from transformer_classifier import *
import time
import os
from collections import defaultdict
import torch.nn.functional as F

print(torch.cuda.is_available())
torch.cuda.empty_cache()
torch.manual_seed(42);


def label(x):
    if x == "cs.AI":
        return 0
    elif x == "cs.NE":
        return 1
    elif x == "th.AC":
        return 2
    elif x == "th.GR":
        return 3


def read_articles():
    train_df = pd.read_csv('../input/articles/train.csv')
    val_df = pd.read_csv('../input/articles/val.csv')
    test_df = pd.read_csv('../input/articles/test.csv')
    # Convert sentiment columns to numerical values
    train_df.label = train_df.label.apply(lambda x: label(x))
    val_df.label = val_df.label.apply(lambda x: label(x))
    test_df.label = test_df.label.apply(lambda x: label(x))
    print(train_df.tail(5), train_df.shape)
    return train_df, val_df, test_df


def load_glove():
    glove = pd.read_csv('../input/glove.6B.100d.txt', sep=" ", quoting=3,
                        header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    # Check check the dimension of this fasttext embedding version
    print(glove_embedding['hello'].shape)
    return glove_embedding


class ArticleDataset:
    def __init__(self, content, label):
        """
        Argument:
        content: a numpy array
        labels: a vector array
        Return xtrain and ylabel in torch tensor datatype, stored in dictionary format
        """
        self.content = content
        self.label = F.one_hot(torch.tensor(label), num_classes=4)

    def __len__(self):
        # return length of dataset
        return len(self.content)

    def __getitem__(self, index):
        # given an idex (item), return content and target of that index in torch tensor
        content = torch.tensor(self.content[index, :], dtype=torch.long)
        label = self.label[index, :]

        return {'content': content,
                'labels': label}


def train(
        data_loader,
        model,
        criterion,
        optimizer,
        device,
        n_class,
        n_examlple):
    """
    this is model training for one epoch
    data_loader:  this is torch dataloader, just like dataset but in torch and devide into batches
    model :
    optimizer : torch optimizer : adam
    device:  cuda or cpu
    """
    # set model to training mode
    model.train()
    # go through batches of data in data loader
    train_loss = 0.0
    final_predictions = []
    final_labels = []
    #text = []
    for data in data_loader:
        contents = data['content']
        labels = data['labels']
        # move the data to device that we want to use
        contents = contents.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)
        # clear the gradient
        optimizer.zero_grad()
        # make prediction from model
        predictions = model(contents)
        #print(predictions.shape, labels.shape)
        # caculate the losses
        loss = criterion(predictions, labels.view(-1, n_class))
        # backprob
        loss.backward()
        train_loss += loss
        #single optimization step
        optimizer.step()
        predictions = predictions.cpu().detach().numpy().tolist()
        labels = data['labels'].cpu().detach().numpy().tolist()
        final_predictions.extend(predictions)
        final_labels.extend(labels)
        #text.extend(data['content'].detach().cpu().numpy())
    # calculate metrics values
    _, train_labels = torch.max(torch.tensor(final_labels), dim=1)
    _, train_outputs = torch.max(torch.tensor(final_predictions), dim=1)
    train_acc = metrics.accuracy_score(train_outputs, train_labels)

    return train_acc, train_loss/n_examlple


def evaluate(
        data_loader,
        model,
        criterion,
        optimizer,
        device,
        n_class,
        n_examlple):
    """
    this is model evaluation for one epoch
    data_loader:  torch dataloader, just like dataset but in torch and devide into batches
    optimizer : torch optimizer : adam
    device:  cuda or cpu
    """
    # set model to evaluate mode
    model.eval()
    # go through batches of data in data loader
    final_predictions = []
    final_labels = []
    with torch.no_grad():
        eval_loss = 0.0
        for data in data_loader:
            contents = data['content']
            labels = data['labels']
            # move the data to device that we want to use
            contents = contents.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            # clear the gradient
            optimizer.zero_grad()
            # make prediction from model
            predictions = model(contents)
            # caculate the losses
            loss = criterion(predictions, labels.view(-1, n_class))
            eval_loss += loss
            #single optimization step
            optimizer.step()
            predictions = predictions.cpu().detach().numpy().tolist()
            labels = data['labels'].cpu().detach().numpy().tolist()
            final_predictions.extend(predictions)
            final_labels.extend(labels)
            #text.extend(data['content'].detach().cpu().numpy())
        # calculate metrics values
        _, eval_labels = torch.max(torch.tensor(final_labels), dim=1)
        _, train_outputs = torch.max(torch.tensor(final_predictions), dim=1)
        eval_acc = metrics.accuracy_score(train_outputs, eval_labels)

    return eval_acc, eval_loss/n_examlple



def train_model():
    """
    train a model having the best validation acc
    """
    n_class = 4
    train_df, val_df, test_df = read_articles()
    # STEP 1: tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_df.content.values.tolist())
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    # STEP 2: embedding
    print('Load Glove embedding')
    glove_embedding = load_glove()
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, d_model=100)
    cfg = {
        'ntoken': [embedding_matrix.shape[0]],
        'ninp': [embedding_matrix.shape[1]],
        'nhead': [2],
        'nhid': [64],
        'nlayers': [2],
        'last_hidden': [32],
        'nclass': [4],
        'max_len': [MAX_LEN]
    }
    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(train_df.content.values)
    xval = tokenizer.texts_to_sequences(val_df.content.values)
    xtest = tokenizer.texts_to_sequences(test_df.content.values)

    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xval = tf.keras.preprocessing.sequence.pad_sequences(xval, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training
    train_dataset = ArticleDataset(content=xtrain, label=train_df.label.values)

    # STEP 5: Load dataset to Pytorch DataLoader
    # after we have train_dataset, we create a torch dataloader to load train_dataset class based on specified batch_size
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    # initialize dataset class for validation
    valid_dataset = ArticleDataset(content=xval, label=val_df.label.values)
    val_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)
    test_dataset = ArticleDataset(content=xtest, label=test_df.label.values)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

    # STEP 6: Running
    device = torch.device('cuda')
    # feed embedding matrix to lstm
    model = Classifier(
                   cfg["ntoken"][0],
                   cfg["ninp"][0],
                   cfg["nhead"][0],
                   cfg["nhid"][0],
                   cfg["nlayers"][0],
                   cfg["last_hidden"][0],
                   cfg["nclass"][0],
                   cfg["max_len"][0],
                   )
    # model_glove = LSTM(embedding_matrix)
    print(model)
    # set model to cuda device`
    model.to(device)
    # initialize Adam optimizer
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True], lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    history = defaultdict(list)
    best_accuracy = 0
    tic = time.perf_counter()
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss = train(
            train_data_loader,
            model,
            criterion,
            optimizer,
            device,
            n_class,
            len(train_data_loader)
        )
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = evaluate(
            val_data_loader,
            model,
            criterion,
            optimizer,
            device,
            n_class,
            len(val_data_loader)
        )
        print(f'Val   loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
    toc = time.perf_counter()
    print('Used %.2f seconds' % (toc - tic))
    plot_results(history)
    # evaluate on test data
    model.load_state_dict(torch.load('best_model_state.bin'))
    model = model.to(device)
    print(len(test_data_loader))
    test_acc, _ = evaluate(
        test_data_loader,
        model,
        criterion,
        optimizer,
        device,
        n_class,
        len(test_data_loader)
    )
    print(f'test accuracy {test_acc}')


def create_embedding_matrix(word_index, embedding_dict=None, d_model=100):
    """
     this function create the embedding matrix save in numpy array
    :param word_index: a dictionary with word: index_value
    :param embedding_dict: a dict with word embedding
    :param d_model: the dimension of word pretrained embedding, here I just set to 100, we will define again
    :return a numpy array with embedding vectors for all known words
    """
    embedding_matrix = np.zeros((len(word_index) + 1, d_model))
    ## loop over all the words
    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix


def plot_results(history):
    plt.gca().cla()
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='train accuracy')
    plt.plot(history['val_acc'], label='validation accuracy')
    plt.xlabel('n_epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.ylim([0, 1])
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='train loss')
    plt.plot(history['val_loss'], label='validation loss')
    plt.xlabel('n_epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('smf_academic.png')


MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
if __name__ == '__main__':
    train_model()


