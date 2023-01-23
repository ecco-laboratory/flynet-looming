# %%
# imports per usual
from io import BytesIO
from typing import Any, Callable, List, Optional, Tuple

import h5py
import pandas as pd
import requests
from PIL import Image
from torchvision.datasets import VisionDataset

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
