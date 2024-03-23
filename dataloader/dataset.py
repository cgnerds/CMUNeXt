import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
        img_ext = ".png",
        num_classes = 1,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []
        self.img_ext = img_ext
        self.num_classes = num_classes

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]
        
        # 1. mask-allinone
        # image = cv2.imread(os.path.join(self._base_dir, 'images', case + self.img_ext))
        # mask = cv2.imread(os.path.join(self._base_dir, 'masks', case + self.img_ext), cv2.IMREAD_GRAYSCALE)
        # masks = [(mask == v) for v in range(self.num_classes)]
        # mask = np.stack(masks, axis=-1).astype('float')
        # print(f"mask.shape: {mask.shape}")
        
        # 2. mask-separate
        image = cv2.imread(os.path.join(self._base_dir, 'images', case + self.img_ext))
        label = []
        for i in range(self.num_classes):
            label.append(cv2.imread(os.path.join(self._base_dir, 'masks', str(i), case + self.img_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(label)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        label = augmented['mask']
        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)
        label = label.astype('float32') / 255
        label = label.transpose(2, 0, 1)

        # return image, label, case
        sample = {"image": image, "label": label, "case": case}
        return sample
