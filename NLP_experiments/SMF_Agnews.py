import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
import tensorflow as tf  # we use both tensorflow and pytorch (pytorch for main part) , tensorflow for tokenizer
from SMF_base import *
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from SMF_NLP_competitor import *
import torch.nn.functional as F

torch.manual_seed(42);
MAX_LEN = 32
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
EPOCHS = 30


def read_AGNEWS():
    train_df = pd.read_csv('../input/AG/AG_train.csv')
    test_df = pd.read_csv('../input/AG/AG_test.csv')
    train_df.columns = ["label", "title", "description"]
    test_df.columns = ["label", "title", "description"]
    # Convert sentiment columns to numerical values
    train_df.label = train_df.label.apply(lambda x: x-1)
    test_df.label = test_df.label.apply(lambda x: x-1)
    return train_df, test_df


def load_glove():
    glove = pd.read_csv('../input/glove.6B.100d.txt', sep=" ", quoting=3,
                        header=None, index_col=0)
    glove_embedding = {key: val.values for key, val in glove.T.items()}

    # Check check the dimension of this fasttext embedding version
    print(glove_embedding['hello'].shape)
    return glove_embedding


class AGDataset:
    def __init__(self, reviews, targets):
        """
        Argument:
        reviews: a numpy array
        targets: a vector array

        Return xtrain and ylabel in torch tensor datatype, stored in dictionary format
        """
        self.reviews = reviews
        self.target = F.one_hot(torch.tensor(targets), num_classes=4)

    def __len__(self):
        # return length of dataset
        return len(self.reviews)

    def __getitem__(self, index):
        # given an idex (item), return review and target of that index in torch tensor
        review = torch.tensor(self.reviews[index, :], dtype=torch.long)
        target = self.target[index, :]

        return {'review': review,
                'target': target}


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
    text = []
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
        predictions, W_final = model(reviews)
        #print(predictions.shape, targets.shape)
        # caculate the losses
        loss = criterion(predictions, targets.view(-1, 4))
        # backprob
        loss.backward()
        train_loss += loss
        #single optimization step
        optimizer.step()
        predictions = predictions.cpu().detach().numpy().tolist()
        targets = data['target'].cpu().detach().numpy().tolist()
        final_predictions.extend(predictions)
        final_targets.extend(targets)
        text.extend(data['review'].detach().cpu().numpy())

    return final_predictions, final_targets, train_loss, W_final, text


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
            predictions, Ws = model(reviews)
            # move prediction and target to cpu
            loss = criterion(predictions, targets.view(-1, 4))
            val_loss += loss
            # add predictions to final_prediction
            predictions = predictions.cpu().numpy().tolist()
            targets = data['target'].cpu().numpy().tolist()
            final_predictions.extend(predictions)
            final_targets.extend(targets)

    return final_predictions, final_targets, val_loss


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


def visualization(W_final, texts, epoch, labels, preds):
    #print(Ws[0].shape, Ws[1].shape)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 30), facecolor="w")
    axes.tick_params(axis='both', which='major', labelsize=20)
    input_sentence = texts[1]
    input_sentence = [x for x in input_sentence]
    axes.imshow(W_final[1].detach().cpu().numpy().transpose(), cmap="Blues")
    #axes.set_yticks(np.arange(len(input_sentence)))
    #axes.set_yticklabels(input_sentence, )
    axes.set_xticks(np.arange(len(input_sentence)))
    axes.set_xticklabels(input_sentence)
    axes.set_title(f'epoch_{epoch}_label:{labels[1]}, preds:{int(preds[1])}', fontweight="bold", size=30)
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.savefig(f'./results/W_epoch_{epoch}.png')



def training():
    train_df, valid_df = read_AGNEWS()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_df.description.values.tolist())
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    print('Load Glove embedding')
    glove_embedding = load_glove()
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embedding_dict=glove_embedding, d_model=100)
    print(embedding_matrix.shape)
    cfg = {
        'n_class': [4],
        'n_hidden_f': [20],
        'n_hidden_g': [3],
        'N': [MAX_LEN],
        'd': [100],
        'n_W': [8],
        'with_g': [True],
        'masking': [True],
        'residual': [False]
    }
    # STEP 3: pad sequence
    xtrain = tokenizer.texts_to_sequences(train_df.description.values)
    xtest = tokenizer.texts_to_sequences(valid_df.description.values)

    # zero padding
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=MAX_LEN)

    # STEP 4: initialize dataset class for training
    train_dataset = AGDataset(reviews=xtrain, targets=train_df.label.values)

    # STEP 5: Load dataset to Pytorch DataLoader
    # after we have train_dataset, we create a torch dataloader to load train_dataset class based on specified batch_size
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=1, shuffle=True)
    # initialize dataset class for validation
    valid_dataset = AGDataset(reviews=xtest, targets=valid_df.label.values)
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=1, shuffle=True)

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
    optimizer = torch.optim.Adam([param for param in model_glove.parameters() if param.requires_grad==True], lr=1e-4)

    print('training model')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(EPOCHS):
        # train one epoch
        train_predictions, train_targets, train_loss, W_final, text = train(train_data_loader, model_glove, optimizer, device)
        train_losses.append(train_loss.cpu().detach().numpy())
        # validate
        outputs, targets, val_loss = evaluate(valid_data_loader, model_glove, device)
        #print(outputs)
        #print(targets)
        val_losses.append(val_loss.cpu().detach().numpy())
        # threshold
        _, train_outputs = torch.max(torch.tensor(train_predictions), dim=1)
        _, val_outputs = torch.max(torch.tensor(outputs), dim=1)
        _, train_targets = torch.max(torch.tensor(train_targets), dim=1)
        _, val_targets = torch.max(torch.tensor(targets), dim=1)
        #train_outputs = np.array(train_predictions) >= 0.5
        #val_outputs = np.array(outputs) >= 0.5
        print(train_outputs, train_targets)
        # calculate accuracy
        train_accuracy = metrics.accuracy_score(train_outputs, train_targets)
        val_accuracy = metrics.accuracy_score(val_outputs, val_targets)
        print(f'epoch: {epoch}, val_accuracy_score: {val_accuracy}, val_loss:{val_loss}')
        val_accs.append(val_accuracy)
        train_accs.append(train_accuracy)
        # decode the index to word and visualize the interaction matrix
        if epoch % 3 == 0:
            vis_dict = {
                        'X': text,
                        'W_final': W_final,
                        'labels': train_targets,
                        'preds': train_outputs
                    }
            sentences_last_batch = []
            for t in text:
                sentence = []
                for s in t:
                    if s == 0:
                        sentence.append('0')
                    else:
                        sentence.append(reverse_word_map[s])
                sentences_last_batch.append(sentence)
            visualization(vis_dict['W_final'], sentences_last_batch, epoch, vis_dict['labels'], vis_dict['preds'])

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