"""
Video Face Manipulation Detection Through Ensemble of CNNs

Author Credits:

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini

GitHub: https://github.com/polimi-ispl/icpr2020dfdc
License: GNU GENERAL PUBLIC LICENSE
"""

from typing import List

import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2

from .data import FrameFaceIterableDataset, get_iterative_real_fake_idxs


class FrameFaceTripletIterableDataset(FrameFaceIterableDataset):

    def __init__(
        self,
        roots: List[str],
        dfs: List[pd.DataFrame],
        size: int,
        scale: str,
        num_triplets: int = -1,
        transformer: A.BasicTransform = ToTensorV2(),
        seed: int = None,
    ):
        """

        :param roots: List of root folders for frames cache
        :param dfs: List of DataFrames of cached frames with 'bb' column as array of 4 elements (left,top,right,bottom)
                   and 'label' column
        :param size: face size
        :param num_triplets: number of samples for the dataset
        :param idxs: sampling indexes triplets (each element is a key for anchor, positive, negative)
        :param scale: Rescale the face to the given size, preserving the aspect ratio.
                      If false crop around center to the given size
        :param transformer:
        :param seed:
        """
        super(FrameFaceTripletIterableDataset, self).__init__(
            roots=roots,
            dfs=dfs,
            size=size,
            scale=scale,
            num_samples=num_triplets * 3,
            transformer=transformer,
            seed=seed,
        )

        self.num_triplet_couples = self.num_samples // 6
        self.num_triplets = self.num_triplet_couples * 2
        self.num_samples = self.num_triplets * 3

    def __len__(self):
        return self.num_triplets

    def __iter__(self):
        random_fake_idxs, random_real_idxs = get_iterative_real_fake_idxs(
            df_real=self.df_real,
            df_fake=self.df_fake,
            num_samples=self.num_samples,
            seed0=self.seed0,
        )

        while len(random_fake_idxs) >= 3 and len(random_real_idxs) >= 3:
            a = self._get_face(random_fake_idxs.pop())[0]
            p = self._get_face(random_fake_idxs.pop())[0]
            n = self._get_face(random_real_idxs.pop())[0]
            yield a, p, n

            a = self._get_face(random_real_idxs.pop())[0]
            p = self._get_face(random_real_idxs.pop())[0]
            n = self._get_face(random_fake_idxs.pop())[0]
            yield a, p, n
