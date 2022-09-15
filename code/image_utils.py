# %%
# imports
from io import BytesIO
from typing import Any, Callable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import requests
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
# Let me create myself a goddamn torch Dataset class for NSD

class NSDDataset(VisionDataset):
    """`NSD Stimuli <https://cvnlab.slite.page/p/NKalgWd__F/>` PyTorch-style Dataset.
    
    It requires the hdf5 file containing the NSD stimuli to have been downloaded.

    Args:
        root (string): Folder where hdf5 file is downloaded to.
        annFile (string): Path to CSV annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    
    """

    def __init__(self,
                 root: str,
                 annFile = str,
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None
                 ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        from ast import literal_eval

        # It's allowed to be pandas!
        # convert cropBox so it reads in as a tuple, not a string of the tuple

        self.metadata = pd.read_csv(annFile, converters = {'cropBox': literal_eval})
        self.hdf = h5py.File(self.root + 'nsd_stimuli.hdf5')
        self.ids = self.metadata['Unnamed: 0'].to_list()
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_image(self, id: int) -> Image.Image:

        image = Image.fromarray(self.hdf['imgBrick'][id])

        return image
    
    def _load_target(self, id: int) -> List[Any]:
        return self.metadata.iloc[id].to_dict()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    
    def __len__(self) -> int:
        return (len(self.ids))
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
