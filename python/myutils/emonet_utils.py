# %%
# imports per usual
from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision
from torch import nn
from torchvision.datasets import VisionDataset


# %%
# Helper function so the Dataset returns the supa raw emo class label
# Now must feed in the class list as an arg, in the event that you're using fewer than the full Phil 20
def get_target_emotion_index(target, classes):
    target = classes.index(target['emotion'])
    # return torch.Tensor([target]).to(int)
    return target

# %%
# Instantiate another shitty little LSTM RNN with linear-to-softmax predictor

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size: int=4096,
        hidden_size: int=110,
        num_classes: int=20
    ) -> None:
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        # I guess I could do this to two linear layers and call that bitch a perceptron
        # Not right now though
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            # Need log probabilities for the loss calculator
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # We don't care about the last hidden state
        x, _ = self.lstm(x)
        # Unpack the sequence, and... hopefully this is safe... throw away video lengths
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # Use the batch lengths to index the actual end frames for each video
        # I'm not sure that the last (padded) frame has it carried forward
        x = torch.stack([x[idx, lens[idx]-1, :] for idx in range(lens.size(dim=0))])
        x = self.classifier(x)
 
        return x


# %%
# Instantiate my shitty little GRU RNN with linear-to-softmax predictor

class GRUClassifier(nn.Module):

    def __init__(
        self,
        input_size: int=4096,
        hidden_size: int=110,
        num_classes: int=20,
        bidirectional: bool=False
    ) -> None:
        super(GRUClassifier, self).__init__()

        hidden_size_coef = 2 if bidirectional else 1

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)
        # I guess I could do this to two linear layers and call that bitch a perceptron
        # Not right now though
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_size*hidden_size_coef, out_features=num_classes),
            # Need log probabilities for the loss calculator
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        # We don't care about the last hidden state
        x, _ = self.gru(x)
        # Unpack the sequence, and... hopefully this is safe... throw away video lengths
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        # Use the batch lengths to index the actual end frames for each video
        # I'm not sure that the last (padded) frame has it carried forward
        x = torch.stack([x[idx, lens[idx]-1, :] for idx in range(lens.size(dim=0))])
        x = self.classifier(x)
 
        return x

    
# %%
# Instantiate linear classifier (the original head for headless EmoNet)

class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_size: int=4096,
        num_classes: int=20
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=num_classes),
            # Need log probabilities for the loss calculator
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.classifier(x)
        return x

# %%
# instantiate emonet (from onnx) pytorch class :3

# Can't use torchvision.models.AlexNet()
# because it comes out slightly diff from Matlab. Lol
class EmoNet(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        alexnet_lrn_alpha = 9.999999747378752e-05

        # TODO: still needs the onnx initializer, whatever that thing is
        # Initialized with an empty module?
        self.initializers = nn.Module()
        # initializer has one weight per RGB per pixel
        self.initializers.register_parameter(name='onnx_initializer_0',param=nn.Parameter(torch.ones(torch.Size([1, 3, 227, 227]))))
        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.Conv_0 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        # Adjust each weight independently
        # ReLU contains no learnable parameters, so the positive values aren't scaled and are left as is
        self.Relu_0 = nn.ReLU()
        # Sort of kind of does lateral inhibition
        # Same input/output weight dimensions
        # Just with the values adjusted
        self.LRN_0 = nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1)
        # Pooling operates separately on each convolved feature map
        # Does it cut down the number of nodes or leave it intact?
        # when kernel_size == stride, each weight is included in only one pool
        # thus downsampling by a factor of kernel_size (or stride)
        # In the past, pooling was the "trendy" way of downsampling (to reduce training computations)
        # while retaining some kind of translation invariance for the incoming features
        # but pooling is neither necessary nor sufficient for those two
        self.MaxPool_0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # Do it all again but with different convolutional dimensions
        # channels per "pixel" going up, n "pixels" actually staying the same
        self.Conv_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2)
        self.Relu_1 = nn.ReLU()
        self.LRN_1 = nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1)
        self.MaxPool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        # and again, a few rounds of channels going up, pixels staying the same
        self.Conv_2 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.Relu_2 = nn.ReLU()
        self.Conv_3 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2)
        self.Relu_3 = nn.ReLU()
        self.Conv_4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2)
        self.Relu_4 = nn.ReLU()
        self.MaxPool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.Conv_5 = nn.Conv2d(256, 4096, kernel_size=6, stride=1)
        self.Relu_5 = nn.ReLU()
        # a conv layer becomes fully connected when kernel_size and stride are both 1?
        # I would think it was fully connected once there's only one pixel left...
        # TODO: Confirm that this layer has only one pixel
        self.Conv_6 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1)
        self.Relu_6 = nn.ReLU()
        # Make sure this outputs 20 channels on one pixel
        # AlexNet that comes with torchvision has this as a Linear module, not Conv2d
        # I feel like they are equivalent when the nodes coming in are 1x1px
        # but this feels hackier somehow. Matlab! Dum dum!
        self.Conv_7 = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1)
        self.Flatten_0 = nn.Flatten(start_dim=-3, end_dim=-1), # flatten all except batch and class dims
        self.Softmax_0 = nn.Softmax(dim=-1)
            
        # TODO: What happens when ReLU is inplace vs not (as ONNX import)?
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        # Don't pass through the initializers module because... it doesn't DO anything to the data
        x = x.to(torch.float)
        x = self.Conv_0(x)
        x = self.Relu_0(x)
        x = self.LRN_0(x)
        x = self.MaxPool_0(x)
        x = self.Conv_1(x)
        x = self.Relu_1(x)
        x = self.LRN_1(x)
        x = self.MaxPool_1(x)
        x = self.Conv_2(x)
        x = self.Relu_2(x)
        x = self.Conv_3(x)
        x = self.Relu_3(x)
        x = self.Conv_4(x)
        x = self.Relu_4(x)
        x = self.MaxPool_2(x)
        x = self.Conv_5(x)
        x = self.Relu_5(x)
        x = self.Conv_6(x)
        x = self.Relu_6(x)
        x = self.Conv_7(x)
        x = self.Flatten_0(x)
        x = self.Softmax_0(x)
        return x

# %%
# Pythonic EmoNet (same as above, minus technically unnecessary layers and grouped nicer)
class EmoNetPythonic(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        alexnet_lrn_alpha = 9.999999747378752e-05

        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=6, stride=1),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1),
            nn.Flatten(start_dim=-3, end_dim=-1), # flatten all except batch and class dims
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        x = x.to(torch.float)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.classifier(x)

        return x

# %%
# Loopy headless EmoNet (effectively Alexnet, minus the last conv aka MLP layer, looping over frames in a video)
# Input tensors MUST be converted to float BEFORE passing into the model!!!
# Otherwise converting them to float automatically puts them back on CPU, ugh
class EmoNetHeadlessVideo(nn.Module):
    def __init__(self) -> None:
        # This creates the classes that will come in from the onnx2torch dict
        # Every parameter has to match for it to read in
        # So we need stuff like the weight initializers, which I think don't actually matter for inference
        super().__init__()
        alexnet_lrn_alpha = 9.999999747378752e-05

        # Kernel size is the size of the moving window in square px
        # 3 channels in per pixel (RGB), 96 channels out per conv center (that's a lotta info!)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alexnet_lrn_alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=6, stride=1),
            nn.ReLU()
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        # This is the one that actually EXECUTES the model
        # We _shouldn't_ need to do any elaborate looping in this
        # (No thanks to HHTseng on GitHub, sorry)
        # because feeding in the data attribute of a PackedSequence
        # already lines up with the metadata needed to put the predictions back into sequences
        # using built-in PyTorch utilities
        x = x.to(torch.float)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        # Drop the redundant height/width dimensions now that it's 1x1
        # But keep the batch dimension if it's 1
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # loop over FRAMES
        # Weirdly, I think this works by processing the nth frame from each batch simultaneously

        return x

# %%
# torch Dataset class for Alan's video FRAMES

class Cowen2017FrameDataset(VisionDataset):
    """`Cowen & Keltner (2017) <https://www.pnas.org/doi/full/10.1073/pnas.1702247114>` PyTorch-style Dataset.

    This dataset returns each video frame as a 3D tensor.

    Args:
        root (string): Enclosing folder where videos are located on the local machine.
        annFile (string): Path to directory of metadata/annotation CSVs.
        censorFile (boolean, optional): Censor Alan's "bad" videos? Defaults to True.
        train (boolean, optional): If True, creates dataset from Kragel et al. (2019)'s training set, otherwise
            from the testing set. Defaults to True.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root: str,
                 annPath: str,
                 censor: bool = True,
                 classes: str = None,
                 train: bool = True,
                 device: str = 'cpu',
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        import os

        import pandas as pd

        self.train = train
        self.device = device

        # Read in the Cowen & Keltner top-1 "winning" human emotion classes
        self.labels = pd.read_csv(os.path.join(annPath, 'frame_10fps_metadata.csv'),
                                   index_col=['video', 'frame'])
        self.labels = self.labels[self.labels['emotion'].isin(emonet_output_classes)]
        
        # Can do it this way bc this metadata has it in a column
        self.labels = self.labels.query(f"{'' if self.train else '~'}train")

        # Flexibly subsample emotion classes based on user input
        if classes is not None:
            self.labels = self.labels[self.labels['emotion'].isin(classes)]

        if censor:
            # Can do it this way bc this metadata has it in a column
            self.labels = self.labels.query('~censored')

        # These come out as 2-tuples, mind you
        self.ids = self.labels.index.to_list()
    
    def _load_frame(self, video_id: str, frame_id: str):
        import os

        video = torchvision.io.read_video(os.path.join(self.root, self.labels.loc[video_id, frame_id]['emotion'], video_id),
                                          pts_unit='sec')
        # None of the videos have audio, so discard that from the loaded tuple
        # Also for convenience, discard dict labeling fps so that the videos look like 4D imgs
        # with dims frames x channels x height x width ... which is NOT the default order!
        frames = video[0].permute((0, 3, 1, 2))

        return frames[frame_id]
    
    def _load_target(self, video_id, frame_id) -> List[Any]:
        target = self.labels.loc[video_id, frame_id].to_dict()
        target['video_id'] = video_id
        target['frame_id'] = frame_id
        
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        video_id, frame_id = self.ids[index]
        frame = self._load_frame(video_id, frame_id)
        target = self._load_target(video_id, frame_id)

        if self.transforms is not None:
            frame, target = self.transforms(frame, target)
            
        frame.to(device=self.device)

        if (self.target_transform is not None) & torch.is_tensor(target):
            target.to(device=self.device)

        return frame, target

    def __len__(self) -> int:
        return len(self.ids)


# %%
# Hot and sexy torch Dataset class for Alan's videos

class Cowen2017Dataset(VisionDataset):
    """`Cowen & Keltner (2017) <https://www.pnas.org/doi/full/10.1073/pnas.1702247114>` PyTorch-style Dataset.

    This dataset returns each video as a 4D tensor.

    Args:
        root (string): Enclosing folder where videos are located on the local machine.
        annFile (string): Path to directory of metadata/annotation CSVs.
        censorFile (boolean, optional): Censor Alan's "bad" videos? Defaults to True.
        train (boolean, optional): If True, creates dataset from Kragel et al. (2019)'s training set, otherwise
            from the testing set. Defaults to True.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root: str,
                 annPath: str,
                 censor: bool = True,
                 classes: str = None,
                 train: bool = True,
                 device: str = 'cpu',
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        import os

        import pandas as pd

        self.train = train
        self.device = device

        # Read in the Cowen & Keltner top-1 "winning" human emotion classes
        self.labels = pd.read_csv(os.path.join(annPath, f"{'train' if self.train else 'test'}_video_10fps_ids.csv"),
                                   index_col='video')
        self.labels = self.labels[self.labels['emotion'].isin(emonet_output_classes)]
        
        # Flexibly subsample emotion classes based on user input
        if classes is not None:
            self.labels = self.labels[self.labels['emotion'].isin(classes)]

        if censor:
            # Truly I wish this was in long form but Alan doesn't like tidy data does he
            censored = pd.read_csv(os.path.join(annPath, 'censored_video_ids.csv'))
            # We don't need to see the censored ones! At least I personally don't
            # I guess the model doesn't have feelings
            self.labels = self.labels[~self.labels.index.isin(censored['less.bad'])]
            self.labels = self.labels[~self.labels.index.isin(censored['very.bad'])]

        self.ids = self.labels.index.to_list()
    
    def _load_video(self, id: str):
        import os

        video = torchvision.io.read_video(os.path.join(self.root, self.labels.loc[id]['emotion'], id),
                                          pts_unit='sec')
        # None of the videos have audio, so discard that from the loaded tuple
        # Also for convenience, discard dict labeling fps so that the videos look like 4D imgs
        # with dims frames x channels x height x width ... which is NOT the default order!
        frames = video[0].permute((0, 3, 1, 2))

        # From when I was still trying to output a read_video item
        # video = (frames, video[1], video[2])

        return frames
    
    def _load_target(self, id: str) -> List[Any]:
        target = self.labels.loc[id].to_dict()
        target['id'] = id
        
        return target
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        video = self._load_video(id)
        target = self._load_target(id)

        if self.transforms is not None:
            video, target = self.transforms(video, target)
            
        video.to(device=self.device)

        if self.target_transform is not None:
            target.to(device=self.device)

        return video, target

    def __len__(self) -> int:
        return len(self.ids)

# %%
# Phil's 20 EmoNet output classes as global variable
emonet_output_classes = [
    'Adoration',
    'Aesthetic Appreciation',
    'Amusement',
    'Anxiety',
    'Awe',
    'Boredom',
    'Confusion',
    'Craving',
    'Disgust',
    'Empathic Pain',
    'Entrancement',
    'Excitement',
    'Fear',
    'Horror',
    'Interest',
    'Joy',
    'Romance',
    'Sadness',
    'Sexual Desire',
    'Surprise'
]