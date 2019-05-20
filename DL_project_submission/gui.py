import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
import sys
from PIL import ImageTk, Image
import io
import tkinter
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import Canvas
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, models
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import torch
import os

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
        self.max_seg_length = 20
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
		
    def sample(self, features, h=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        batch_size = features.size(0)
        feature_size = features.size(-1)
        vocab_size = self.vocab_size
        features = features.view(batch_size, -1, feature_size)
        inputs =  self.embedding(torch.tensor([0]))
        if h == None:
            h = self.init_hidden_state(features)

        for i in range(self.max_seg_length):
            attention_weighted_encoding, alpha = self.attention(features,
                                                                h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h = self.decode_step(torch.cat([inputs,attention_weighted_encoding], dim =1), h)
            outputs = self.fc(self.dropout(h))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                              
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)                       # inputs: (batch_size, embed_size)                      
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def get_img_data(filename, maxsize=(1200, 850), first=False):
    img = Image.open(filename)
    img.thumbnail(maxsize)
    if first:
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def pred_window():

    new_window = tkinter.Toplevel(window)
    new_window.geometry("600x450")

    img_path = filedialog.askopenfilename()
    print(img_path)
    #param
    encoder_path = './best_model_encoder.pth'
    decoder_path = './best_model_decoder.pth'
    vocab_path = './vocab.pkl'
    embed_size = 256
    hidden_size = 512
    attention_dim = 512

    # Image preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(attention_dim, embed_size, hidden_size, len(vocab)).to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path,map_location=device))
    decoder.load_state_dict(torch.load(decoder_path,map_location=device))
    encoder.eval()
    decoder.eval()
	

    # Prepare an image
    image = load_image(img_path, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    features = encoder(image_tensor)
    sampled_ids = decoder.sample(features)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    # label showing
    label_show = tkinter.Label(new_window, text=sentence).pack()
    img = ImageTk.PhotoImage(Image.open(img_path))
    loader = tkinter.Label(new_window, image=img)
    loader.image = img
    loader.pack()

    


window = tkinter.Tk()
window.title("GUI")
window.geometry("300x100")
root = tkinter.Frame(window).pack()


btn_img = tkinter.Button(root, text="Predict Single Image", command=pred_window, height=2, width=30).pack()

window.mainloop()





