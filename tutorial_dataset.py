import json
import cv2
import numpy as np

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        with open('/root/STCC/training/new_sen1-2_train_rs-llava.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sar_filename = item['source']
        opt_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/root/autodl-tmp/sen1-2/trainA/' + sar_filename)
        target = cv2.imread('/root/autodl-tmp/sen1-2/trainB/' + opt_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize the images
        clip = cv2.resize(source, (224, 224))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        clip = clip.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, clip=clip, emp="")



class TestDataset(Dataset):
    def __init__(self):
        with open('./testing/new_sar2opt_test_rs-llava.json', 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sar_filename = item['source']
        opt_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('/root/autodl-tmp/sen1-2/testA/' + sar_filename)
        target = cv2.imread('/root/autodl-tmp/sen1-2/testB/' + opt_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Resize the images 
        # source = cv2.resize(source, (224, 224))
        # target = cv2.resize(target, (224, 224))
        clip = cv2.resize(source, (224, 224))

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        clip = clip.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(data=dict(jpg=target, txt=prompt, hint=source, clip=clip, emp=""), filename = opt_filename)

