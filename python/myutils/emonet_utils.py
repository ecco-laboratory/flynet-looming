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
# Pythonic EmoNet (minus technically unnecessary layers from Matlab-to-ONNX and grouped using nn.Sequential)
class EmoNet(nn.Module):
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

        import pandas as pd

        self.device = device

        # Read in the Cowen & Keltner top-1 "winning" human emotion classes
        self.labels = pd.read_csv(annPath,
                                  index_col='video')
        # Keep only the 20 used in Kragel et al. 2019, defined in this module
        self.labels = self.labels[self.labels['emotion'].isin(emonet_output_classes)]

        # Choose training or testing split
        if train:
            self.labels = self.labels[self.labels['split'] == 'train']
        else:
            self.labels = self.labels[self.labels['split'] == 'test']

        # Flexibly subsample emotion classes based on user input
        if classes is not None:
            self.labels = self.labels[self.labels['emotion'].isin(classes)]

        if censor:
            # We don't need to see the censored ones! At least I personally don't
            # I guess the model doesn't have feelings
            self.labels = self.labels[~self.labels['censored']]

        self.ids = self.labels.index.to_list()
    
    def _load_video(self, id: str):
        import os

        video = torchvision.io.read_video(os.path.join(self.root, id),
                                          pts_unit='sec')
        # None of the videos have audio, so discard that from the loaded tuple
        # Also for convenience, discard dict labeling fps so that the videos look like 4D imgs
        # with dims frames x channels x height x width ... which is NOT the default order!
        frames = video[0].permute((0, 3, 1, 2))

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