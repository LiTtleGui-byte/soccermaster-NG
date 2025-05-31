import os
import copy
import random
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
import torch

from data.utils import Compose, ToTensor, RandomResize, Normalize, get_image_hw

role_mapping = {'ball': 0, 'goalkeeper': 1, 'other': 2, 'player': 3, 'referee': 4, None: -1}
reid_columns = ["role", "team", "jersey_number"]

class SoccerNetGSR_ReID(Dataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "SN-GSR-2024",
            split: str = "train",
            transforms=None,
    ):
        super(SoccerNetGSR_ReID, self).__init__()
        assert split == 'train', "only train split is supported yet"
        self.split = split
        self.transforms = transforms
        
        self.column_mapping = {}
        self.reid_columns = reid_columns
        self.role_mapping = role_mapping
        self.data_dir = os.path.join(data_root, sub_dir)
        # Load dataframes from pickle files
        train_df = pd.read_pickle(os.path.join(self.data_dir, 'ReID_df', 'train_df.pkl'))
        query_df = pd.read_pickle(os.path.join(self.data_dir, 'ReID_df', 'query_df.pkl'))
        gallery_df = pd.read_pickle(os.path.join(self.data_dir, 'ReID_df', 'gallery_df.pkl'))

        train, query, gallery = self.to_torchreid_dataset_format(
            [train_df, query_df, gallery_df]
        )
        self.train = train
        
        # self.query = query
        # self.gallery = gallery
        # {'pid': 0, 'camid': '060', 'img_path': '/GPFS/rhome/haolinyang/sports/soccernet/sn-gamestate/datasets/SN-GSR-2024/SoccerNetGS/reid/images/train/060/0_060_1060000001.jpg', 'masks_path': '', 'visibility': 1, 'image_id': '1060000001', 'video_id': '060', 'role': 3, 'team': 0, 'jersey_number': 0}
        
    def __len__(self):
        return len(self.train)
    
    def __getitem__(self, index):
        return self.train[index]
        
    def to_torchreid_dataset_format(self, dataframes):
        results = []
        column_mapping = {}
        column_mapping["role"] = self.role_mapping
        for col in self.reid_columns:
            if col not in column_mapping:
                unique_values = {element for df in dataframes for element in df[col].unique()}
                unique_values.discard(None)
                ordered_unique_values = list(unique_values)
                ordered_unique_values.sort()
                column_mapping[col] = {
                    v: i for i, v in enumerate(ordered_unique_values)
                }
                column_mapping[col][None] = -1

        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df["camid"] = df["video_id"]
            df["img_path"] = df["reid_crop_path"]
            # remove bbox_head as it is not available for each sample
            # df to list of dict
            sorted_df = df.sort_values(by=["pid"])
            # use only necessary annotations: using them all caused a
            # 'RuntimeError: torch.cat(): input types can't be cast to the desired output type Long' in collate.py
            # -> still has to be fixed
            data_list = sorted_df[
                ["pid", "camid", "img_path", "masks_path", "visibility", "image_id", "video_id"] + self.reid_columns
            ].copy()  # create a copy to avoid SettingWithCopyWarning
            
            # factorize all columns, i.e. replace string values with 0-based increasing ids
            for col in self.reid_columns:
                data_list.loc[:, col] = data_list[col].map(column_mapping[col])  # 使用 .loc 进行赋值
                self.column_mapping[col] = {value: key for key, value in column_mapping[col].items()}

            data_list = data_list.to_dict("records")
            results.append(data_list)
        return results
    
    def __getitem__(self, index):
        sample = self.train[index]
        image_path = sample['img_path']
        image = Image.open(image_path).convert("RGB")
            
        annotation = {
            'pid': sample['pid'],
            'visibility': sample['visibility'],
            'role': sample['role'],
            'team': sample['team'],
            'jersey_number': sample['jersey_number'],
        }
        
        metas = {
            'task': 'SoccerNetGSR_ReID',
            'split': self.split,
        }
        
        if self.transforms is not None:
            image, annotation, metas = self.transforms(image, annotation, metas)
        
        return image, annotation, metas

def build_gsr_reid_dataset(config: dict, split: str):
    dataset = SoccerNetGSR_ReID(
        data_root=config["DATA_ROOT"],
        sub_dir=config["SoccerNetGSR_SUB_DIR"],
        split=split,
        transforms=build_transforms(config),
    )
    return dataset

def build_gsr_reid_dataloader(config: dict, split: str):
    dataset = build_gsr_reid_dataset(config, split)
    sampler = PrtreidSampler(dataset.train, batch_size=config["SOCCER_NET_GSR_REID_BATCH_SIZE"], num_instances=config["SOCCER_NET_GSR_REID_NUM_INSTANCES"], column_mapping=dataset.column_mapping)
    return DataLoader(dataset, batch_size=config["SOCCER_NET_GSR_REID_BATCH_SIZE"], collate_fn=collate_fn, num_workers=config["NUM_WORKERS"], sampler=sampler)

class PrtreidSampler(Sampler):
    """Samples for all three tasks: reid, role, and team

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances, column_mapping):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        assert batch_size % 32 == 0, "batch_size must be divisible by 32"
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.column_mapping = column_mapping
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        self.game_dic = {}
        for index, sample in enumerate(self.data_source):
            self.index_dic[sample['pid']].append(index)
            if sample['video_id'] not in self.game_dic.keys():
                self.game_dic[sample['video_id']] = {'left': defaultdict(list), 'right': defaultdict(list), 'other': defaultdict(list)}
            if self.column_mapping['role'][sample['role']] == 'player':  # goalkeeper not part of team classification
                team = self.column_mapping['team'][sample['team']]
                self.game_dic[sample['video_id']][team][sample['pid']].append(index)
            else:
                self.game_dic[sample['video_id']]['other'][sample['pid']].append(index)

        self.pids = list(self.index_dic.keys())
        self.gids = list(self.game_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        batch_games_dic = copy.deepcopy(self.game_dic)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_gids = copy.deepcopy(self.gids)
        final_idxs = []

        while len(avai_gids) > 0:
            selected_game = random.sample(avai_gids, 1)[0]
            ################### from left side ############################
            avai_pids = copy.deepcopy([i for i in batch_games_dic[selected_game]['left'].keys()])
            selected_pids = random.sample(avai_pids, 3)
            ################### from right side ###########################
            avai_pids = copy.deepcopy([i for i in batch_games_dic[selected_game]['right'].keys()])
            selected_pids += random.sample(avai_pids, 3)
            ################## from other roles ###########################
            avai_pids = copy.deepcopy([i for i in batch_games_dic[selected_game]['other'].keys()])
            selected_pids += random.sample(avai_pids, 2)

            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                if pid in batch_games_dic[selected_game]['other']:
                    batch_idxs_dict[pid].append(batch_idxs)

                final_idxs.extend(batch_idxs)

                if len(batch_idxs_dict[pid]) == 0:
                    del batch_idxs_dict[pid]
                    if 'left' in batch_games_dic[selected_game].keys() and pid in batch_games_dic[selected_game]['left']:
                        del batch_games_dic[selected_game]['left'][pid]
                        if len(batch_games_dic[selected_game]['left']) < 3:
                            del batch_games_dic[selected_game]['left']

                    elif 'right' in batch_games_dic[selected_game].keys() and pid in batch_games_dic[selected_game]['right']:
                        del batch_games_dic[selected_game]['right'][pid]
                        if len(batch_games_dic[selected_game]['right']) < 3:
                            del batch_games_dic[selected_game]['right']

                    elif 'other' in batch_games_dic[selected_game].keys() and pid in batch_games_dic[selected_game]['other']:
                        del batch_games_dic[selected_game]['other'][pid]
                        if len(batch_games_dic[selected_game]['other']) < 2:
                            del batch_games_dic[selected_game]['other']

            if len(batch_games_dic[selected_game].keys()) < 3:
                    avai_gids.remove(selected_game)

        return iter(final_idxs)

    def __len__(self):
        return self.length
    
def build_transforms(config: dict):

    return Compose([
        ToTensor(),
        RandomResize(sizes=config["AUG_RANDOM_RESIZE"], max_size=config["AUG_MAX_SIZE"], keep_aspect_ratio=config["KEEP_ASPECT_RATIO"]),
        Normalize(mean=config["AUG_MEAN"], std=config["AUG_STD"]),
    ])
    
def collate_fn(batch):
    images, annotations, metas = zip(*batch)
    _B = len(batch)
    images = torch.stack(images)
    annotations = {k: torch.tensor([anno[k] for anno in annotations]) for k in annotations[0]}

    return {
        "images": images,
        "annotations": annotations,
        "metas": metas,
    }