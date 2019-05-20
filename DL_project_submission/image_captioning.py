import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import nltk
nltk.download('punkt')
from PIL import Image
from pycocotools.coco import COCO
from collections import Counter
import torch.nn.functional as F
import time
from nltk.translate.bleu_score import corpus_bleu

# from google.colab import drive
# drive.mount('/content/drive')
# root = '/content/drive/My Drive/dlp'

root = os.path.dirname(os.path.abspath(__file__))
print("Current directory:", root)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =================================
# --------------------- Build_Vocab ---------------------
# =================================
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
      
    def get_word(self, index):
        return self.idx2word[index]

def build_vocab(json, threshold):
    """Build a simple vocabulary wrapper."""
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

# =================================
# --------------------- Dataloader ---------------------  
# =================================
class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
# =================================
# ------------------------ model ------------------------
# =================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, encoded_image_size = 14):
        """Load the pretrained ResNet-18 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        model = models.densenet121(pretrained=False)
        modules = list(model.children())[:-1]  # delete the last fc layer.
        self.model = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        """Extract feature vectors from input images."""
        # train the model
        features = self.model(images) # (batch_size, 1024, 7, 7)
        features = self.adaptive_pool(features) # (batch_size, 1024, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 1024)

        return features


class Attention(nn.Module):
    """
    Attention Network.        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        feature = self.bn(self.linear(out))
    """

    def __init__(self, feature_size, hidden_size, attention_dim):
        """
        :param feature_size: feature size of encoded images
        :param hidden_size: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(feature_size, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(hidden_size, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, features, hidden_size):
        """
        Forward propagation.
        :param features: encoded images, a tensor of dimension (batch_size, num_pixels, feature_size)
        :param hidden_size: previous decoder output, a tensor of dimension (batch_size, hidden_size)
        :return: attention weighted encoding, weights
        """

        att1 = self.encoder_att(features)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(hidden_size)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (features * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, feature_size)

        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_size, hidden_size, vocab_size, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_size: embedding size
        :param hidden_size: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param feature_size: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderRNN, self).__init__()
        feature_size = 1024
        self.feature_size = feature_size
        self.attention_dim = attention_dim
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(feature_size, hidden_size, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_size)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.GRUCell(embed_size + feature_size, hidden_size, bias=True)  # decoding GRUCell
        self.init_h = nn.Linear(feature_size, hidden_size)  # linear layer to find initial hidden state of GRUCell
        self.f_beta = nn.Linear(hidden_size, feature_size)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, features):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param features: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = features.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_size)
        return h

    def forward(self, features, captions, lengths):
        """
        Forward propagation.
        :param features: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = features.size(0)
        feature_size = features.size(-1)
        vocab_size = self.vocab_size
        # Flatten image
        features = features.view(batch_size, -1, feature_size)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = features.size(1)
        lengths = torch.tensor(lengths)

        # Sort input data by decreasing lengths; why? apparent below
        lengths, sort_ind = lengths.sort(dim=0, descending=True)
        features = features[sort_ind]
        captions = captions[sort_ind]

        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize GRU state
        h = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(features[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                h[:batch_size_t])  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, sort_ind
# =================================
# ------------------------- Train -------------------------
# =================================
def train(encoder, decoder, train_loader, epoch, num_epochs, criterion, optimizer, log_step, save_step, model_path):
    print("------------------ Start training ---------------")
    # Train the models
    encoder.train()
    decoder.train()
    t_loss = 0
    total_step = len(train_loader)
    print(len(train_loader.dataset))
    start = time.time()
    for i, (images, captions, lengths) in enumerate(train_loader):

        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)

        # Forward, backward and optimize
        features = encoder(images)
    
        if i % 100 == 0:
            print(i, "/",total_step)
        outputs,caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, captions, lengths)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        outputs, _ = pack_padded_sequence(outputs, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        loss = criterion(outputs, targets)
        t_loss += loss.item()
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        # Print log info
        if i % log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}, Time: {:.2f} mins'
                  .format(epoch+1, num_epochs, i, total_step, loss.item(), np.exp(loss.item()),(time.time() - start)/60))

            # Save the model checkpoints
        if (i + 1) % save_step == 0:
            torch.save(decoder.state_dict(), os.path.join(
                model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
            torch.save(encoder.state_dict(), os.path.join(
                model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))

    t_loss /= len(train_loader.dataset)
    return t_loss
# =================================
# ---------------------Validation ---------------------
# =================================
def val(encoder, decoder, val_loader, criterion,vocab):
    print("------------------ Start validating ---------------")
    # validate the models
    encoder.eval()
    decoder.eval()
    val_loss = 0
    correct = 0
    total_step = len(val_loader)
    print(len(val_loader.dataset))
    s = 0

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    start = time.time()
    with torch.no_grad():
        for i, (images, captions, lengths) in enumerate(val_loader):
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)

            # Forward, backward and optimize
            features = encoder(images)
            outputs, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, captions, lengths)

            targets = caps_sorted[:, 1:]
            opt_copy = outputs.clone()
            outputs, _ = pack_padded_sequence(outputs, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss_function = nn.CrossEntropyLoss(reduction='sum')
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(targets.view_as(pred)).sum().item()
            s += len(pred)

            for ele in captions:             
                elist = ele.tolist()
                sentenselist = []
                for w in elist:
                    if w not in [0,1]:
                        sentenselist.append(w)
                templist = []
                templist.append(sentenselist)
                references.append(templist)
               

            temp_preds = list()
            _, preds = torch.max(opt_copy, dim=2)
            preds = preds.tolist()
            for index, ele in enumerate(preds):
                temp_preds.append(preds[index][:decode_lengths[index]])  # remove pad
            preds = temp_preds
            hypotheses.extend(preds)

            if i % 1000 == 0:
                assert len(references) == len(hypotheses)
                bleu4 = corpus_bleu(references, hypotheses)
                print('Step [{}/{}], Loss: {:.4f} , Accuracy: {:.4f}, BLEU score: {:.4f}, Time: {:.2f} mins'
                      .format(i, total_step, loss.item(), correct / s, bleu4, (time.time() - start) /60 ))
                
                gt = references[i][0]
                pt = hypotheses[i]
                
                gt_str = ""
                pt_str = ""
                for ind in range(len(gt)):
                    gt_str+=vocab.get_word(gt[ind]) +" "
                    pt_str+=vocab.get_word(pt[ind]) +" "
                    
                print("target sentense: ", gt_str)
                print("pred sentense: ", pt_str)

    bleu4 = corpus_bleu(references, hypotheses)

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), BLEU score: {:.4f},Time: {:.2f} mins\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset), bleu4, (time.time() - start)/60 ))

    val_accuracy = correct / len(val_loader.dataset)
    return val_loss, val_accuracy, bleu4
# =================================
# ------------------------- Main -----------------------  
# =================================
def main(model_path, crop_size, vocab_path, image_dir_train, image_dir_val, caption_path_train, caption_path_val,
         log_step, save_step, embed_size, hidden_size,
         num_epochs, batch_size, num_workers, learning_rate, attention_dim):
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Image preprocessing, normalization for the pretrained densenet
    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader for train and val
    train_loader = get_loader(image_dir_train, caption_path_train, vocab,
                              transform, batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = get_loader(image_dir_val, caption_path_val, vocab,
                            transform, batch_size,
                            shuffle=True, num_workers=num_workers)

    # Build the models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(attention_dim, embed_size, hidden_size, len(vocab)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters()) # train all layers in encoder and decoder parameters
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    best_epoch = 0
    best_accuracy = 0
    best_model = None
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    start = time.time()
    # train
    for epoch in range(num_epochs):
        train_loss = train(encoder, decoder, train_loader, epoch, num_epochs, criterion, optimizer, log_step, save_step,
                           model_path)
        print("Total time taken for train at epoch " + str(epoch), (time.time() - start) / 60)
        torch.save(encoder.state_dict(), root + '/model_encoder' + str(epoch) + '.pth')
        torch.save(decoder.state_dict(), root + '/model_decoder' + str(epoch) + '.pth')
        
        val_loss, val_accuracy, bleu4 = val(encoder, decoder, val_loader, criterion,vocab)
        print("Time", time.time() - start)
        
#         train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_accuracy)
        if val_accuracy > best_accuracy:
            best_epoch = epoch + 1
            best_accuracy = val_accuracy
            best_model_en = encoder.state_dict()
            best_model_de = decoder.state_dict()

    print("Best_accuracy: ", best_accuracy)
    encoder.load_state_dict(best_model_en)
    decoder.load_state_dict(best_model_de)
    torch.save(encoder.state_dict(), root + '/best_model_encoder' + '.pth')
    torch.save(decoder.state_dict(), root + '/best_model_decoder' + '.pth')


# =================================
# -------------------------- Run ------------------------
# =================================
if __name__ == '__main__':
    image_dir_train = root + '/data/resized2014'
    image_dir_val = root + '/data/resized_val2014'
    caption_path_train = root + '/data/annotations/captions_train2014.json'
    caption_path_val = root + '/data/annotations/captions_val2014.json'
    
    model_path = root + '/data/models/'
    vocab_path = root + '/vocab.pkl'
    
    crop_size = 224
    
    log_step = 100
    save_step = 10000

    # Model parameters
    embed_size = 256
    hidden_size = 512

    num_epochs = 3
    batch_size = 32
    num_workers = 2
    learning_rate = 0.001

    attention_dim = 512

    main(model_path, crop_size, vocab_path, image_dir_train, image_dir_val, caption_path_train, caption_path_val,
         log_step, save_step, embed_size, hidden_size,
         num_epochs, batch_size, num_workers, learning_rate,attention_dim)