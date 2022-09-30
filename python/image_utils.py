# %%
# imports
from io import BytesIO
from typing import Any, Callable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import requests
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset

# %%
# read_and_preprocess_image

def read_and_preprocess_image(path, cropbox = None):
    if path.startswith('http'):
        # use requests and io.BytesIO to pull from web
        img = Image.open(BytesIO(requests.get(path).content))
    else:
        img = Image.open(path)
    
    # expect pre-formatted pillow cropbox
    if cropbox is not None:
        img = img.crop(cropbox)
    # resize and then render as numpy array
    # but leave the image in 0-255 RGB space as that's also how Matlab imread returns imgs
    # TODO: soft-code the dimensions to get the height and width of the img from the model
    img = np.asarray(img.resize((227, 227)))
    # move the RGB axis to from last to first
    # TODO: Figure out why the matlab code was reversing all the axes, not just the last one
    img = np.moveaxis(img, -1, 0)

    return img

# %%
# cropbox_nsd_to_pillow
def cropbox_nsd_to_pillow(size, cropbox):
    # expect size to come in from the COCO metadata
    # expect cropbox to come in from the NSD stim dataframe

    if sum(cropbox[0:2]) == 0:
        upper = 0
        left = size[0] * cropbox[2]
        lower = size[1]
        right = size[0] - (size[0] * cropbox[3])
    else:
        upper = size[1] * cropbox[0]
        left = 0
        lower = size[1] - (size[1] * cropbox[1])
        right = size[0]
    
    return (left, upper, right, lower)

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
        these_videos = pd.read_csv(os.path.join(annPath, f"{'train' if self.train else 'test'}_video_ids.csv"),
                                   index_col='video')

        # Read in the Cowen & Keltner metadata

        self.metadata = pd.read_csv(os.path.join(annPath, 'video_ratings.csv'),
                                    index_col='Filename')
        
        # Just the emotion categories
        self.metadata = self.metadata.iloc[:, range(34)]

        # attach the pre-computed "winning" class labels
        # Include only the designated training or testing set
        # Using join as an implicit filter kills 2 tasks with one command
        self.metadata = self.metadata.join(these_videos, how='right')
        
        if censor:
            # Truly I wish this was in long form but Alan doesn't like tidy data does he
            censored = pd.read_csv(os.path.join(annPath, 'censored_video_ids.csv'))
            # We don't need to see the censored ones! At least I personally don't
            # I guess the model doesn't have feelings
            self.metadata = self.metadata[~self.metadata.index.isin(censored['less.bad'])]
            self.metadata = self.metadata[~self.metadata.index.isin(censored['very.bad'])]

        self.ids = self.metadata.index.to_list()
    
    def _load_video(self, id: str):
        import os

        video = torchvision.io.read_video(os.path.join(self.root, self.metadata.loc[id]['emotion'], id),
                                          pts_unit='sec')
        # None of the videos have audio, so discard that from the loaded tuple
        # Also for convenience, discard dict labeling fps so that the videos look like 4D imgs
        video = video[0]
        # with dims frames x channels x height x width ... which is NOT the default order!
        video = video.permute((0, 3, 1, 2))

        return video
    
    def _load_target(self, id: str) -> List[Any]:
        target = self.metadata.loc[id].to_dict()
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

        return image
    
    def _load_target(self, id: int) -> List[Any]:
        return self.metadata.loc[id].to_dict()

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
