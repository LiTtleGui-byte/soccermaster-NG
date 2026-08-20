import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import zipfile
import pickle
import copy
from PIL import Image
from soccermaster.data.pnlcalib_utils.utils_keypoints import KeypointsDB
from soccermaster.data.pnlcalib_utils.utils_lines import LineKeypointsDB
from tqdm import tqdm
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def correct_lines_labels(data):
    """修正lines标签的函数，参考soccernet_gsr_detection.py"""
    if 'Goal left post left' in data.keys():
        data['Goal left post left '] = copy.deepcopy(data['Goal left post left'])
        del data['Goal left post left']
    return data

if __name__ == "__main__":
    # 设置数据路径
    data = {'Goal left post left': {'x': 100, 'y': 100}, 'Goal left post right': {'x': 200, 'y': 200}}
    correct_lines_labels(data)
    print(data)