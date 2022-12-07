# %%
# imports up top, babies

import os
import platform
from functools import partial

import pandas as pd
import torch
from emonet_utils import (Cowen2017FrameDataset, EmoNetHeadlessVideo,
                          LinearClassifier, emonet_output_classes,
                          get_target_emotion_index)
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

# Set this to emonet_output_classes if using all of them
these_classes = ['Anxiety', 'Excitement', 'Fear', 'Horror', 'Surprise']

k19_train = pd.read_csv(os.path.join(local_ck_path, 'metadata', 'frame_10fps_metadata.csv'), index_col=['video', 'frame'])
k19_train = k19_train.query('train')
# Keep only the ones from the 20 classes that Phil kept
k19_train = k19_train[k19_train['emotion'].isin(emonet_output_classes)]

# Get the relative frequencies in alphabetical order
emonet_class_weights = k19_train['emotion'].value_counts(normalize=True).sort_index()
emonet_class_weights = [round(weight, 4) for weight in emonet_class_weights.to_list()]
emonet_class_weights = torch.Tensor(emonet_class_weights)
# Literally every tensor needs to be moved to GPU dear god
emonet_class_weights = emonet_class_weights.to(device)

# We need this later to reverse-weight the training sampler
# To choose fewer goddamn amusement videos! My goodness
emonet_class_weights_reverse = k19_train['emotion'].value_counts().sort_index()
emonet_class_weights_reverse = [round(emonet_class_weights_reverse.sum()/weight, 4) for weight in emonet_class_weights_reverse]

# Separate set of weights when only including these four emotion classes
# Again, one set for weighting the classes in the loss function
# And one somewhat reciprocal-ed set for oversampling
# Remember, the classes are _alphabetized!_
emonet_nclass_weights = k19_train['emotion'][k19_train['emotion'].isin(these_classes)].value_counts(normalize=True).sort_index()
emonet_nclass_weights = [round(weight, 4) for weight in emonet_nclass_weights.to_list()]
emonet_nclass_weights = torch.Tensor(emonet_nclass_weights)
# Literally every tensor needs to be moved to GPU dear god
emonet_nclass_weights = emonet_nclass_weights.to(device)

emonet_nclass_weights_reverse = k19_train['emotion'][k19_train['emotion'].isin(these_classes)].value_counts().sort_index()
emonet_nclass_weights_reverse = [round(emonet_nclass_weights_reverse.sum()/weight, 4) for weight in emonet_nclass_weights_reverse]

# %%
# Fire up AlexNet base
emonet_headless = EmoNetHeadlessVideo()
emonet_headless.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'EmoNetHeadless.pt')))

# Frozen! No training!
for param in emonet_headless.parameters():
    param.requires_grad = False

emonet_headless.to(device)
# %%
# Fire up a new linear head
classifier = LinearClassifier(num_classes=len(these_classes))
classifier.to(device)

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
these_target_transforms = partial(get_target_emotion_index, classes=these_classes)

ck_train = Cowen2017FrameDataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    classes=these_classes,
    train=True,
    device=device,
    transform=these_transforms,
    # A single vector-tensor of indices is what the loss function expects
    target_transform=these_target_transforms
)

ck_test = Cowen2017FrameDataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    classes=these_classes,
    train=False,
    device=device,
    transform=these_transforms,
    target_transform=these_target_transforms
)

ck_test_unaugmented = Cowen2017FrameDataset(
    root=video_path,
    annPath=metadata_path,
    censor=False,
    classes=these_classes,
    train=False,
    device='cpu',
    transform=transforms.Resize((227, 227))
)
# Mind you: because this routes the videos through pad_sequence() in order to batch them,
# it also returns a list of the nframes of each video in the batch
# Phil used minibatches of 16 for the EmoNet paper
# But... we have so few goddamn videos

oversampler = torch.utils.data.WeightedRandomSampler(
    weights=emonet_nclass_weights_reverse,
    num_samples=len(ck_train)
)

batch_size = 16
ck_train_torchloader = torch.utils.data.DataLoader(
    ck_train,
    batch_size=batch_size,
    shuffle=True
)
ck_test_torchloader = torch.utils.data.DataLoader(
    ck_test,
    batch_size=batch_size,
    shuffle=True
)

# %%
# Define training and testing epoch functions

loss_fn_unweighted = torch.nn.NLLLoss()
loss_fn_weighted = torch.nn.NLLLoss(weight=emonet_nclass_weights)
# Phil's initial EmoNet learning rate was 1e-4 but I feel like that shit ain't enough
learning_rate = 1e-3
# Spoopy! We're gonna use stochastic gradient descent bc Phil used it for EmoNet
optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

def train_decoder(dataloader, encoder, decoder, loss_fn, optimizer):
    
    with tqdm(dataloader) as progress_dataloader:
        progress_dataloader.set_description('Training')

        for frames, targets in progress_dataloader:
            frames = frames.to(torch.float)
            frames = frames.to(device)
            # Tensors can't modify device in place
            # (although models can, confusing)
            targets = targets.to(device)

            # Remember! This model is frozen!
            encoded = encoder(frames)

            # The actual model prediction that is getting trained
            preds = decoder(encoded)
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
        
        for batch, (frames, targets) in enumerate(progress_dataloader):
            frames = frames.to(torch.float)
            frames = frames.to(device)
            targets = targets.to(device)

            # Remember! This model is frozen!
            encoded = encoder(frames)

            preds = decoder(encoded)

            loss += loss_fn(preds, targets)
            avg_loss = loss/(batch+1)

            correct += (preds.argmax(1) == targets).type(torch.float).sum().item()
            accuracy = correct/((batch+1)*batch_size)

            # Print the loss off somewhere
            progress_dataloader.set_postfix(accuracy=accuracy, avg_loss=avg_loss)

# %%
# Run the training! Agh!
# By default models are initialized in training mode
# So we only need to explicitly call this for the frozen CNN
emonet_headless.eval()

n_epochs = 5

for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_decoder(
        ck_train_torchloader,
        emonet_headless,
        classifier,
        loss_fn_weighted,
        optimizer
    )
    test_decoder(
        ck_test_torchloader,
        emonet_headless,
        classifier,
        loss_fn_weighted
    )
print('By god, you\'ve done it.')
torch.save(classifier.state_dict(), '../ignore/models/LinearClassifier20221014_01.pt')
# %%
# Just getting the predictions on the test data
classifier = LinearClassifier(num_classes=len(these_classes))
classifier.load_state_dict(state_dict=torch.load(os.path.join(model_path, 'LinearClassifier20221014_01.pt'), map_location=torch.device(device)))
classifier.to(device)
classifier.eval()
video_ids = []
frame_ids = []
preds_all = []

for frame, target in tqdm(ck_test_unaugmented):
    frame = frame.to(torch.float)
    frame = frame.to(device)

    # Remember! This model is frozen!
    encoded = emonet_headless(frame)

    # Now this model is also frozen! Predictions only
    with torch.no_grad():
        preds = classifier(encoded)
    
    preds = preds.to('cpu')
    video_ids.append(target['video_id'])
    frame_ids.append(target['frame_id'])
    preds_all.append(preds.numpy())

preds_all = [pd.DataFrame(ps, columns = these_classes) for ps in preds_all]

preds_all = pd.concat(preds_all)
preds_all.index = pd.MultiIndex.from_arrays(arrays=[video_ids, frame_ids], names=['video', 'frame'])

preds_all['guess_1'] = preds_all.idxmax(axis=1)

preds_all.to_csv(os.path.join(metadata_path, 'test_linear_5class_preds.csv'))

# %%
