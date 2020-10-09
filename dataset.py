import os
import cv2
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from randomcrop import RandomRotation,RandomResizedCrop,RandomHorizontallyFlip,RandomVerticallyFlip
from PIL import Image, ImageFile
from skimage import io

import pickle
import random

random.seed(2020)

train_root = '/media/zjnu/Local Disk/Training Datasets/Real_Rain_Streaks_Dataset_CVPR19/training'
test_root = '/media/zjnu/Local Disk/Training Datasets/Real_Rain_Streaks_Dataset_CVPR19/testing'

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.dataset = name
        self.mat_files = []
        with open(name,'rb') as f:
            mat_files = pickle.load(f)
        for key in mat_files:
            if len(mat_files[key]) > 1: 
                self.mat_files.append({key: mat_files[key]})
        self.file_num = len(self.mat_files)
        self.rc = RandomResizedCrop(256)

    def __len__(self):
        return self.file_num * 100

    def __getitem__(self, idx):
        while True:
            try:
                file_name = self.mat_files[idx % self.file_num]
                size_samples = len(list(file_name.values())[0])
                rand_1 = random.randint(0, size_samples - 1)
                rand_2 = random.randint(0, size_samples - 1)
                while rand_2 == rand_1:
                    rand_2 = random.randint(0, size_samples - 1)
                gt_file = file_name[list(file_name)[0]][rand_1]
                img_file = file_name[list(file_name)[0]][rand_2]

                # Image.open(train_root + img_file).verify()
                # io.imread(train_root + img_file)  

                O = cv2.imread(train_root + img_file)
                B = cv2.imread(train_root + gt_file)

                O = Image.fromarray(O)
                B = Image.fromarray(B)

                O, B = self.rc(O,B)
                O, B = np.array(O), np.array(B)

                M = np.clip((O-B).sum(axis=2),0,1).astype(np.float32)
                O = np.transpose(O.astype(np.float32) / 255, (2, 0, 1))
                B = np.transpose(B.astype(np.float32) / 255, (2, 0, 1))

                sample = {'O': O, 'B': B,'M':M}

                return sample
            except:
                print('detect error img %s' % img_file)
                idx = random.randint(0, self.file_num)



class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = name
        self.mat_files = open(self.root_dir,'r').readlines()

        self.file_num = len(self.mat_files)
        
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]

        gt_file = file_name.split(' ')[1][:-1]
        img_file = file_name.split(' ')[0]
        
        O = cv2.imread(test_root + img_file)
        B = cv2.imread(test_root + gt_file)

        O = np.transpose(O, (2, 0, 1)).astype(np.float32) / 255.0 
        B = np.transpose(B, (2, 0, 1)).astype(np.float32) / 255.0 

        sample = {'O': O,'B':B,'M':O}

        return sample

    # def verify_image(img_file):
    #     try:
    #         img = io.imread(img_file)
    #     except:
    #         return False
    #     return True

# if __name__ == "__main__":
#     rainy_dir = {}
#     text_dir = '/media/zjnu/Local Disk/Training Datasets/Real_Rain_Streaks_Dataset_CVPR19/training/real_world_orig.txt'
#     dir_root = '/media/zjnu/Local Disk/Training Datasets/Real_Rain_Streaks_Dataset_CVPR19/training'
#     mat_files = open(text_dir,'r').readlines()
#     progress = 0
#     for file in mat_files:
#         gt_file = file.split(' ')[1][:-1]
#         img_file = file.split(' ')[0]
#         if os.path.isfile(dir_root + img_file):
#             try:
#                 Image.open(dir_root + img_file).verify()
#                 cv2.imread(train_root + img_file)
#                 io.imread(train_root + img_file)            
#                 if gt_file in rainy_dir:
#                     rainy_dir[gt_file].append(img_file)
#                 else:
#                     rainy_dir[gt_file] = [img_file]
#             except:
#                 # os.remove(dir_root + img_file) 
#                 print('detect error img %s' % dir_root + img_file)
#                 continue
#         progress += 1
#         if progress % 1000 == 0:
#             print(progress * 100 / len(mat_files))
#     with open('/media/zjnu/Local Disk/Training Datasets/Real_Rain_Streaks_Dataset_CVPR19/training/real_world.pkl', 'wb') as f:
#         pickle.dump(rainy_dir, f)
