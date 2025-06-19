from typing import Dict, List, Tuple

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
License: GNU GENERAL PUBLIC LICENSE 3.0
"""
import numpy as np
import pandas as pd

available_datasets = [
    "dfdc-35-5-10",
    "ff-c23-720-140-140",
    "ff-c23-720-140-140-5fpv",
    "ff-c23-720-140-140-10fpv",
    "ff-c23-720-140-140-15fpv",
    "ff-c23-720-140-140-20fpv",
    "ff-c23-720-140-140-25fpv",
    "celebdf",  # just for convenience, not used in the original paper
]


def load_df(
    dfdc_df_path: str,
    ffpp_df_path: str,
    dfdc_faces_dir: str,
    ffpp_faces_dir: str,
    dataset: str,
) -> (pd.DataFrame, str):
    if dataset.startswith("dfdc"):
        df = pd.read_pickle(dfdc_df_path)
        root = dfdc_faces_dir
    elif dataset.startswith("ff-"):
        df = pd.read_pickle(ffpp_df_path)
        root = ffpp_faces_dir
    else:
        raise NotImplementedError("Unknown dataset: {}".format(dataset))
    return df, root


def get_split_df(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    if dataset == "dfdc-35-5-10":
        if split == "train":
            split_df = df[df["folder"].isin(range(35))]
        elif split == "val":
            split_df = df[df["folder"].isin(range(35, 40))]
        elif split == "test":
            split_df = df[df["folder"].isin(range(40, 50))]
        else:
            raise NotImplementedError("Unknown split: {}".format(split))
    elif dataset.startswith("ff-c23-720-140-140"):
        # Save random state
        st0 = np.random.get_state()
        # Set seed for this selection only
        np.random.seed(41)
        # Split on original videos
        crf = dataset.split("-")[1]
        random_youtube_videos = np.random.permutation(
            df[(df["source"] == "youtube") & (df["quality"] == crf)]["video"].unique()
        )
        train_orig = random_youtube_videos[:720]
        val_orig = random_youtube_videos[720 : 720 + 140]
        test_orig = random_youtube_videos[720 + 140 :]
        if split == "train":
            split_df = pd.concat(
                (df[df["original"].isin(train_orig)], df[df["video"].isin(train_orig)]),
                axis=0,
            )
        elif split == "val":
            split_df = pd.concat(
                (df[df["original"].isin(val_orig)], df[df["video"].isin(val_orig)]),
                axis=0,
            )
        elif split == "test":
            split_df = pd.concat(
                (df[df["original"].isin(test_orig)], df[df["video"].isin(test_orig)]),
                axis=0,
            )
        else:
            raise NotImplementedError("Unknown split: {}".format(split))

        if dataset.endswith("fpv"):
            fpv = int(dataset.rsplit("-", 1)[1][:-3])
            idxs = []
            for video in split_df["video"].unique():
                idxs.append(
                    np.random.choice(
                        split_df[split_df["video"] == video].index, fpv, replace=False
                    )
                )
            idxs = np.concatenate(idxs)
            split_df = split_df.loc[idxs]
        # Restore random state
        np.random.set_state(st0)
    elif dataset == "celebdf":

        seed = 41
        num_real_train = 600

        # Save random state
        st0 = np.random.get_state()
        # Set seed for this selection only
        np.random.seed(seed)
        # Split on original videos
        random_train_val_real_videos = np.random.permutation(
            df[(df["label"] == False) & (df["test"] == False)]["video"].unique()
        )
        train_orig = random_train_val_real_videos[:num_real_train]
        val_orig = random_train_val_real_videos[num_real_train:]
        if split == "train":
            split_df = pd.concat(
                (df[df["original"].isin(train_orig)], df[df["video"].isin(train_orig)]),
                axis=0,
            )
        elif split == "val":
            split_df = pd.concat(
                (df[df["original"].isin(val_orig)], df[df["video"].isin(val_orig)]),
                axis=0,
            )
        elif split == "test":
            split_df = df[df["test"] == True]
        else:
            raise NotImplementedError("Unknown split: {}".format(split))
        # Restore random state
        np.random.set_state(st0)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(dataset))
    return split_df


def make_splits(
    dfdc_df: str, ffpp_df: str, dfdc_dir: str, ffpp_dir: str, dbs: Dict[str, List[str]]
) -> Dict[str, Dict[str, Tuple[pd.DataFrame, str]]]:
    """
    Make split and return Dataframe and root
    :param
    dfdc_df: str, path to the DataFrame containing info on the faces extracted from the DFDC dataset with extract_faces.py
    ffpp_df: str, path to the DataFrame containing info on the faces extracted from the FF++ dataset with extract_faces.py
    dfdc_dir: str, path to the directory containing the faces extracted from the DFDC dataset with extract_faces.py
    ffpp_dir: str, path to the directory containing the faces extracted from the FF++ dataset with extract_faces.py
    dbs: {split_name:[split_dataset1,split_dataset2,...]}
                Example:
                {'train':['dfdc-35-5-15',],'val':['dfdc-35-5-15',]}
    :return: split_dict: dictonary containing {split_name: ['train', 'val'], splitdb: List(pandas.DataFrame, str)}
                Example:
                {'train, 'dfdc-35-5-15': (dfdc_train_df, 'path/to/dir/of/DFDC/faces')}
    """
    split_dict = {}
    full_dfs = {}
    for split_name, split_dbs in dbs.items():
        split_dict[split_name] = dict()
        for split_db in split_dbs:
            if split_db not in full_dfs:
                full_dfs[split_db] = load_df(
                    dfdc_df, ffpp_df, dfdc_dir, ffpp_dir, split_db
                )
            full_df, root = full_dfs[split_db]
            split_df = get_split_df(df=full_df, dataset=split_db, split=split_name)
            split_dict[split_name][split_db] = (split_df, root)

    return split_dict
