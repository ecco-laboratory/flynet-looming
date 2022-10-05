# %%
# imports per usual
from typing import Any, Callable, List, Optional, Tuple, Union

import h5py
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.datasets import VisionDataset

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
        self.Flatten_0 = nn.Flatten()
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
            nn.Flatten(),
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
                 train: bool = True,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)

        import os

        import pandas as pd

        self.train = train

        # Read in the Cowen & Keltner top-1 "winning" human emotion classes
        self.labels = pd.read_csv(os.path.join(annPath, f"{'train' if self.train else 'test'}_video_10fps_ids.csv"),
                                   index_col='video')
        
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

        return video, target

    def __len__(self) -> int:
        return len(self.ids)

# %%
# Let me create myself a goddamn torch Dataset class for NSD

class NSDDataset(VisionDataset):
    """`NSD Stimuli <https://cvnlab.slite.page/p/NKalgWd__F/>` PyTorch-style Dataset.
    
    It requires the hdf5 file containing the NSD stimuli to have been downloaded.

    Args:
        root (string): Folder where hdf5 file is downloaded to.
        annFile (string): Path to CSV annotation file.
        shared1000 (boolean): Run using the subset of stimuli that all participants saw? Defaults to False (all 73000 possible stimuli).
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    
    """

    def __init__(self,
                 root: str,
                 annFile: str,
                 shared1000: bool = False,
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None
                 ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        from ast import literal_eval

        # It's allowed to be pandas!
        # convert cropBox so it reads in as a tuple, not a string of the tuple

        self.metadata = pd.read_csv(annFile, converters = {'cropBox': literal_eval})
        self.metadata = self.metadata.set_index('Unnamed: 0')
        if shared1000:
            self.metadata = self.metadata.query('shared1000')
        self.hdf = h5py.File(self.root + 'nsd_stimuli.hdf5')
        self.ids = self.metadata.index.to_list()
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_image(self, id: int) -> Image.Image:
        image = Image.fromarray(self.hdf['imgBrick'][id])

        if self.transform is not None:
            image = self.transform(image)

        return image
    
    def _load_target(self, id: int) -> List[Any]:
        target = self.metadata.loc[id].to_dict()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    
    def __len__(self) -> int:
        return len(self.ids)
# %%
# torchvision cocodetection but it reads from web (cursed?)

class CocoDetectionFromWeb(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        cocoAnnFile: str,
        nsdAnnFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(cocoAnnFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["coco_url"]
        return Image.open(BytesIO(requests.get(path).content)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self) -> int:
        return len(self.ids)
