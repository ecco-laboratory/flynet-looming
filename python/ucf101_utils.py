# %%
# imports per usual

import os
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

# %%
# Instantiate UCF101 _frame_ classes

# Per Simonyan & Zisserman 2014, RGB takes a single random frame from each clip
# Flow takes from the selected frame and the 9 after it, for 20 channels (x, y times 10 frames)
# Clips are NOT farmed out into individual frames, ahem Phil
class UCF101TwoStreamDataset(VisionDataset):
    """`UCF-101 2-stream <https://proceedings.neurips.cc/paper/2014/file/00ec53c4682d36f5c4359f4ae7bd7ba1-Paper.pdf>` PyTorch-style Dataset.

    This dataset returns a _2-tuple_ with a single RGB frame and a 10-forward-stacked xy flow-frame as 3D tensors.

    Args:
        root (string): Enclosing folder where videos are located on the local machine.
        annPath (string): Path to directory of metadata/annotation CSVs.
        train (boolean, optional): If True, creates dataset from Yi Zhu's pytorch/GitHub training set,
            otherwise from the testing set. Defaults to True.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    def __init__(
        self,
        root: str,
        annPath: str,
        train: bool = True,
        split: int = 1,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        import pandas as pd

        self.classes = pd.read_csv(
            os.path.join(annPath, 'classInd.txt'),
            sep = ' ',
            names = ['class_id', 'class_name']
        )
        # need this one from UCF (I think) to start indexing at 0
        # because the ones from the repo people start at 0
        self.classes['class_id'] = self.classes['class_id']-1
        
        self.classes = self.classes.set_index('class_id')

        self.train = train

        if self.train:
            # Fine to use just the rgb split because the files are the same apparently
            listname = 'train_rgb_split{:01d}.txt'.format(split)
        else:
            listname = 'val_rgb_split{:01d}.txt'.format(split)

        self.labels = pd.read_csv(
            os.path.join(annPath, listname),
            sep = ' ',
            names = ['filename', 'nframes', 'class']
        )

    def _load_rgb(self, video_id: int, frame_id: int):
        out = Image.open(
            os.path.join(
                self.root,
                'images',
                self.classes.loc[self.labels['class'][video_id]]['class_name'],
                self.labels['filename'][video_id]+'_f{:04d}.jpeg'.format(frame_id)
            )
        )

        return transforms.functional.to_tensor(out)
    
    def _load_flow(self, video_id: int, frame_id: int):
        
        out = []
        for current_frame in range(frame_id, frame_id+10):
            current_out = Image.open(
                os.path.join(
                    self.root,
                    'flow',
                    self.classes.loc[self.labels['class'][video_id]]['class_name'],
                    self.labels['filename'][video_id]+'_f{:04d}.jpeg'.format(current_frame)
                )
            )
            current_out = np.array(current_out)
            # drop the empty green dimension
            # in ndarrays, the channel dimension is LAST
            current_out = current_out[..., [0, 2]]
            out.append(np.array(current_out))
        
        out = np.concatenate(out, axis=2)

        return transforms.functional.to_tensor(out)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        import random

        target = self.labels['class'][index]

        # Do not allow choosing from the last 10 frames
        # as those won't have 10 subsequent frames of flow
        frame_id = random.randint(0, self.labels['nframes'][index] - 10)
        
        rgb = self._load_rgb(index, frame_id)
        flow = self._load_flow(index, frame_id)

        if self.transform is not None:
            rgb = self.transform(rgb)
            flow = self.transform(flow)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return (rgb, flow, target)
    
    def __len__(self) -> int:
        return len(self.labels)
