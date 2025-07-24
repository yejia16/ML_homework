import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 加载所有图片路径和ID
        images_df = pd.read_csv(os.path.join(self.root_dir, 'images.txt'),
                                sep=' ', names=['img_id', 'filepath'])

        # 加载图片ID和类别标签
        labels_df = pd.read_csv(os.path.join(self.root_dir, 'image_class_labels.txt'),
                                sep=' ', names=['img_id', 'label'])

        # 加载训练/测试集划分信息
        split_df = pd.read_csv(os.path.join(self.root_dir, 'train_test_split.txt'),
                               sep=' ', names=['img_id', 'is_train'])

        # 合并所有信息到一个DataFrame
        data_df = images_df.merge(labels_df, on='img_id').merge(split_df, on='img_id')

        # 根据 train 参数选择训练集或测试集
        if train:
            self.data = data_df[data_df['is_train'] == 1].copy()
        else:
            self.data = data_df[data_df['is_train'] == 0].copy()

        # 标签是从1开始的，我们需要把它变成从0开始
        self.data['label'] = self.data['label'] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 获取图片路径和标签
        img_path = os.path.join(self.root_dir, 'images', self.data.iloc[idx]['filepath'])
        label = self.data.iloc[idx]['label']

        # 打开图片
        image = Image.open(img_path).convert('RGB')

        # 应用数据变换
        if self.transform:
            image = self.transform(image)

        return image, label
