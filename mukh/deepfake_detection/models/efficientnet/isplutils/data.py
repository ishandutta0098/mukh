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

import os
from pathlib import Path
from typing import List

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from .utils import extract_bb


def load_face(
    record: pd.Series, root: str, size: int, scale: str, transformer: A.BasicTransform
) -> torch.Tensor:
    path = os.path.join(str(root), str(record.name))
    autocache = size < 256 or scale == "tight"
    if scale in [
        "crop",
        "scale",
    ]:
        cached_path = str(
            Path(root)
            .joinpath("autocache", scale, str(size), str(record.name))
            .with_suffix(".jpg")
        )
    else:
        # when self.scale == 'tight' the extracted face is not dependent on size
        cached_path = str(
            Path(root)
            .joinpath("autocache", scale, str(record.name))
            .with_suffix(".jpg")
        )

    face = np.zeros((size, size, 3), dtype=np.uint8)
    if os.path.exists(cached_path):
        try:
            face = Image.open(cached_path)
            face = np.array(face)
            if len(face.shape) != 3:
                raise RuntimeError("Incorrect format: {}".format(path))
        except KeyboardInterrupt as e:
            # We want keybord interrupts to be propagated
            raise e
        except (OSError, IOError) as e:
            print("Deleting corrupted cache file: {}".format(cached_path))
            print(e)
            os.unlink(cached_path)
            face = np.zeros((size, size, 3), dtype=np.uint8)

    if not os.path.exists(cached_path):
        try:
            frame = Image.open(path)
            bb = record["left"], record["top"], record["right"], record["bottom"]
            face = extract_bb(frame, bb=bb, size=size, scale=scale)

            if autocache:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                face.save(cached_path, quality=95, subsampling="4:4:4")

            face = np.array(face)
            if len(face.shape) != 3:
                raise RuntimeError("Incorrect format: {}".format(path))
        except KeyboardInterrupt as e:
            # We want keybord interrupts to be propagated
            raise e
        except (OSError, IOError) as e:
            print("Error while reading: {}".format(path))
            print(e)
            face = np.zeros((size, size, 3), dtype=np.uint8)

    face = transformer(image=face)["image"]

    return face


class FrameFaceIterableDataset(IterableDataset):

    def __init__(
        self,
        roots: List[str],
        dfs: List[pd.DataFrame],
        size: int,
        scale: str,
        num_samples: int = -1,
        transformer: A.BasicTransform = ToTensorV2(),
        output_index: bool = False,
        labels_map: dict = None,
        seed: int = None,
    ):
        """

        :param roots: List of root folders for frames cache
        :param dfs: List of DataFrames of cached frames with 'bb' column as array of 4 elements (left,top,right,bottom)
                   and 'label' column
        :param size: face size
        :param num_samples:
        :param scale: Rescale the face to the given size, preserving the aspect ratio.
                      If false crop around center to the given size
        :param transformer:
        :param output_index: enable output of df_frames index
        :param labels_map: map from 'REAL' and 'FAKE' to actual labels
        """

        self.dfs = dfs
        self.size = int(size)

        self.seed0 = int(seed) if seed is not None else np.random.choice(2**32)

        # adapt indices
        dfs_adapted = [df.copy() for df in self.dfs]
        for df_idx, df in enumerate(dfs_adapted):
            mi = pd.MultiIndex.from_tuples(
                [(df_idx, key) for key in df.index], names=["df_idx", "df_key"]
            )
            df.index = mi
        # Concat
        self.df = pd.concat(dfs_adapted, axis=0, join="inner")

        self.df_real = self.df[self.df["label"] == 0]
        self.df_fake = self.df[self.df["label"] == 1]

        self.longer_set = "real" if len(self.df_real) > len(self.df_fake) else "fake"
        self.num_samples = max(len(self.df_real), len(self.df_fake)) * 2
        self.num_samples = (
            min(self.num_samples, num_samples) if num_samples > 0 else self.num_samples
        )

        self.output_idx = bool(output_index)

        self.scale = str(scale)
        self.roots = [str(r) for r in roots]
        self.transformer = transformer

        self.labels_map = labels_map
        if self.labels_map is None:
            self.labels_map = {
                False: np.array(
                    [
                        0.0,
                    ]
                ),
                True: np.array(
                    [
                        1.0,
                    ]
                ),
            }
        else:
            self.labels_map = dict(self.labels_map)

    def _get_face(
        self, item: pd.Index
    ) -> (torch.Tensor, torch.Tensor) or (torch.Tensor, torch.Tensor, str):

        record = self.dfs[item[0]].loc[item[1]]
        face = load_face(
            record=record,
            root=self.roots[item[0]],
            size=self.size,
            scale=self.scale,
            transformer=self.transformer,
        )

        label = self.labels_map[record.label]
        if self.output_idx:
            return face, label, record.name
        else:
            return face, label

    def __len__(self):
        return self.num_samples

    def __iter__(self):

        random_fake_idxs, random_real_idxs = get_iterative_real_fake_idxs(
            df_real=self.df_real,
            df_fake=self.df_fake,
            num_samples=self.num_samples,
            seed0=self.seed0,
        )

        while len(random_fake_idxs) >= 1 and len(random_real_idxs) >= 1:
            yield self._get_face(random_fake_idxs.pop())
            yield self._get_face(random_real_idxs.pop())


def get_iterative_real_fake_idxs(
    df_real: pd.DataFrame, df_fake: pd.DataFrame, num_samples: int, seed0: int
):
    longer_set = "real" if len(df_real) > len(df_fake) else "fake"
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        seed = seed0
        np.random.seed(seed)
        worker_num_couple_samples = num_samples // 2
        fake_idxs_portion = np.random.choice(
            df_fake.index, worker_num_couple_samples, replace=longer_set == "real"
        )
        real_idxs_portion = np.random.choice(
            df_real.index, worker_num_couple_samples, replace=longer_set == "fake"
        )
    else:
        worker_id = worker_info.id
        seed = seed0 + worker_id
        np.random.seed(seed)
        worker_num_couple_samples = (num_samples // 2) // worker_info.num_workers
        if longer_set == "fake":
            fake_idxs_portion = df_fake.index[
                worker_id
                * worker_num_couple_samples : (worker_id + 1)
                * worker_num_couple_samples
            ]
            real_idxs_portion = np.random.choice(
                df_real.index, worker_num_couple_samples, replace=True
            )
        else:
            real_idxs_portion = df_real.index[
                worker_id
                * worker_num_couple_samples : (worker_id + 1)
                * worker_num_couple_samples
            ]
            fake_idxs_portion = np.random.choice(
                df_fake.index, worker_num_couple_samples, replace=True
            )
    random_fake_idxs = list(np.random.permutation(fake_idxs_portion))
    random_real_idxs = list(np.random.permutation(real_idxs_portion))

    assert len(random_fake_idxs) == len(random_real_idxs)

    return random_fake_idxs, random_real_idxs


class FrameFaceDatasetTest(Dataset):

    def __init__(
        self,
        root: str,
        df: pd.DataFrame,
        size: int,
        scale: str,
        transformer: A.BasicTransform = ToTensorV2(),
        labels_map: dict = None,
        aug_transformers: List[A.BasicTransform] = None,
    ):
        """

        :param root: root folder for frames cache
        :param df: DataFrame of cached frames with 'bb' column as array of 4 elements (left,top,right,bottom)
                   and 'label' column
        :param size: face size
        :param num_samples:
        :param scale: Rescale the face to the given size, preserving the aspect ratio.
                      If false crop around center to the given size
        :param transformer:
        :param labels_map: dcit to map df labels
        :param aug_transformers: if not None, creates multiple copies of the same sample according to the provided augmentations
        """

        self.df = df
        self.size = int(size)

        self.scale = str(scale)
        self.root = str(root)
        self.transformer = transformer
        self.aug_transformers = aug_transformers

        self.labels_map = labels_map
        if self.labels_map is None:
            self.labels_map = {
                False: np.array(
                    [
                        0.0,
                    ]
                ),
                True: np.array(
                    [
                        1.0,
                    ]
                ),
            }
        else:
            self.labels_map = dict(self.labels_map)

    def _get_face(
        self, item: pd.Index
    ) -> (torch.Tensor, torch.Tensor) or (torch.Tensor, torch.Tensor, str):
        record = self.df.loc[item]
        label = self.labels_map[record.label]
        if self.aug_transformers is None:
            face = load_face(
                record=record,
                root=self.root,
                size=self.size,
                scale=self.scale,
                transformer=self.transformer,
            )
            return face, label
        else:
            faces = []
            for aug_transf in self.aug_transformers:
                faces.append(
                    load_face(
                        record=record,
                        root=self.root,
                        size=self.size,
                        scale=self.scale,
                        transformer=A.Compose([aug_transf, self.transformer]),
                    )
                )
            faces = torch.stack(faces)
            return faces, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self._get_face(self.df.index[item])
