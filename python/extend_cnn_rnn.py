# %%
# imports up top, babies

import os
import platform

import pandas as pd
import torch
from emonet_utils import (Cowen2017Dataset, EmoNet, EmoNetHeadlessVideo,
                          GRUClassifier, emonet_output_classes,
                          get_target_emotion_index)
from image_utils import pad_sequence_tuple
from torch.nn.utils import rnn as rnn_utils
from torchvision import transforms
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# Set abs paths based on which cluster node we're on
base_path = 'data/eccolab'
if platform.node() != 'ecco':
    base_path = os.path.join(os.sep, 'home', base_path)
else:
    base_path = os.path.join(os.sep, base_path)

model_path = '../ignore/models'
local_ck_path = '/home/mthieu/Repos/CowenKeltner'
# %%
# Calculate weights for emo classes bc they are... unbalanced
# I'm looking at you, amusement

k19_train = pd.read_csv(os.path.join(local_ck_path, 'metadata', 'train_video_ids.csv'), index_col='video')
# Keep only the ones from the 20 classes that Phil kept
k19_train = k19_train[k19_train['emotion'].isin(emonet_output_classes)]
# Get the relative frequencies in alphabetical order
emonet_class_weights = k19_train['emotion'].value_counts(normalize=True).sort_index()
# Roudning to 4 digits made them add up to 1. Deal with it
emonet_class_weights = [round(weight, 4) for weight in emonet_class_weights.to_list()]

# %%
# Fire up a newer net
emonet_headless = EmoNetHeadlessVideo()
emonet_headless.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'EmoNetHeadless.pt')))

for param in emonet_headless.parameters():
    param.requires_grad = False

emonet_headless.to(device)
# %%
# Fire up a recurrent net!
decoder_gru = GRUClassifier()
decoder_gru.to(device)
# %%
# Load ze data

# Augmentation transforms to try:
# Horizontal flip
# Grayscale?
# Color jitter?
# Look at the augmentations done on, e.g., Kinetics

video_path = os.path.join(local_ck_path, 'videos_10fps')
metadata_path = os.path.join(local_ck_path, 'metadata')
these_transforms = transforms.Compose([
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((227, 227))
])
these_target_transforms = get_target_emotion_index

ck_train = Cowen2017Dataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    train=True,
    transform=these_transforms,
    # A single vector-tensor of indices is what the loss function expects
    target_transform=these_target_transforms
)

ck_test = Cowen2017Dataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    train=False,
    transform=these_transforms,
    target_transform=these_target_transforms
)

# Mind you: because this routes the videos through pad_sequence() in order to batch them,
# it also returns a list of the nframes of each video in the batch
# Phil used minibatches of 16 for the EmoNet paper
# But... we have so few goddamn videos

batch_size = 8
ck_train_torchloader = torch.utils.data.DataLoader(
    ck_train,
    batch_size=batch_size,
    collate_fn=pad_sequence_tuple
)
ck_test_torchloader = torch.utils.data.DataLoader(
    ck_test,
    batch_size=batch_size,
    collate_fn=pad_sequence_tuple
)
# %%
# Define training and testing epoch functions

loss_fn = torch.nn.NLLLoss(weight=emonet_class_weights)
# Phil's initial EmoNet learning rate was...
learning_rate = 1e-4
# Why not? From the PyTorch 'optimizing parameters' tutorial
n_epochs = 5
# Spoopy! We're gonna use stochastic gradient descent bc Phil used it for EmoNet
optimizer = torch.optim.SGD(decoder_gru.parameters(), lr=learning_rate)

def train_decoder(dataloader, encoder, decoder, loss_fn, optimizer):
    
    with tqdm(dataloader) as progress_dataloader:
        progress_dataloader.set_description('Training')

        for vids, lens, targets in progress_dataloader:
            vids_packed = rnn_utils.pack_padded_sequence(
                input=vids,
                lengths=lens,
                batch_first=True,
                enforce_sorted=False
            )

            # Remember! This model is frozen!
            encoded = encoder(vids_packed.data)
            encoded_packed = rnn_utils.PackedSequence(
                data=encoded,
                batch_sizes=vids_packed.batch_sizes,
                sorted_indices=vids_packed.sorted_indices,
                unsorted_indices=vids_packed.unsorted_indices
            )

            # The actual model prediction that is getting trained
            preds = decoder(encoded_packed)
            loss = loss_fn(preds, targets)

            # Backpropagation... pytorch makes it look so smooth
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print the loss off somewhere
            progress_dataloader.set_postfix(loss=loss.item())

# By decorating on top of the whole test function definition
# Nothing inside should have gradients set
@torch.no_grad()
def test_decoder(dataloader, encoder, decoder, loss_fn):
    loss, correct = 0, 0
    batch_size = dataloader.batch_size

    with tqdm(dataloader) as progress_dataloader:
        progress_dataloader.set_description('Testing')
        
        for batch, (vids, lens, targets) in enumerate(progress_dataloader):
            vids_packed = rnn_utils.pack_padded_sequence(
                input=vids,
                lengths=lens,
                batch_first=True,
                enforce_sorted=False
            )

            # Remember! This model is frozen!
            encoded = encoder(vids_packed.data)
            encoded_packed = rnn_utils.PackedSequence(
                data=encoded,
                batch_sizes=vids_packed.batch_sizes,
                sorted_indices=vids_packed.sorted_indices,
                unsorted_indices=vids_packed.unsorted_indices
            )

            preds = decoder(encoded_packed)

            loss += loss_fn(preds, targets)
            avg_loss = loss/batch

            correct += (preds.argmax(1) == targets).type(torch.float).sum().item()
            accuracy = correct/(batch*batch_size)

            # Print the loss off somewhere
            progress_dataloader.set_postfix(accuracy=accuracy, avg_loss=avg_loss)

# %%
# Run the training! Agh!
# By default models are initialized in training mode
# So we only need to explicitly call this for the frozen CNN
emonet_headless.eval()

for epoch in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_decoder(
        ck_train_torchloader,
        emonet_headless,
        decoder_gru,
        loss_fn,
        optimizer
    )
    test_decoder(
        ck_test_torchloader,
        emonet_headless,
        decoder_gru,
        loss_fn
    )
print('By god, you\'ve done it.')
# %%
# Testing out the GRU output

