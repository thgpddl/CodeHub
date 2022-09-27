import time

from torch.utils.data import Dataset, DataLoader
import os
import torch
from torchvision import transforms
from PIL import Image


def _find_classes(set_path):
    """

    Args:
        set_path:

    Returns:
        classes=["file_name_1","file_name_2","file_name_3",...]
        class_to_idx={'file_name_1': 0,
                      'file_name_2': 1,
                      'file_name_3': 2,
                       ...}

    """
    classes = [d.name for d in os.scandir(set_path) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def _make_dataset(dir, class_to_idx):
    """
    将dir下所有类文件夹下所有图片路径和类打包为元组，并添加到images
    Args:
        dir: 类文件夹上级目录，即dir下就是类文件夹
        class_to_idx: {"subfile_name":int}

    Returns: images

    """
    print("image数据加载中...")
    images = []

    for target in sorted(class_to_idx.keys()):  # 所有子文件夹名
        d = os.path.join(dir, target)  # 拼接完整子文件夹路径
        if not os.path.isdir(d):
            continue

        # os.walk返回：正在遍历的这个文件夹的本身的地址、所有子文件夹名、所以子文件名（非递归）
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)  # 拼接完整文件路径
                item = (path, class_to_idx[target])  # 将img路径和label打包为元组
                images.append(item)

    return images


class Animals10(Dataset):
    def __init__(self, set_path, transform, preread=False):
        super(Animals10, self).__init__()
        self.transform = transform
        self.preread=preread
        classes, class_to_idx = _find_classes(set_path)
        self.images = _make_dataset(set_path, class_to_idx)  # 每个元素是元组(path,label)

        if self.preread: # 将全部图片读进内存
            self.images_var=[]
            for path, _ in self.images:
                img = Image.open(path).convert('RGB')
                self.images_var.append(img)



    def __getitem__(self, index):
        """

        Args:
            index:

        Returns: (input,target)

        """
        path, target = self.images[index]

        if self.preread:
            img = self.images_var[index]    # 直接从内存取
        else:
            img = Image.open(path).convert('RGB')   # 需要自己读

        img = self.transform(img)
        target = torch.tensor(target, dtype=torch.int64)

        # img：float32   target：int64
        return img, target

    def __len__(self):
        return self.images.__len__()


def get_train_loader(data_root_path, batch_size, workers,preread):
    train_transform = transforms.Compose([
        transforms.Resize(size=(255, 255)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = Animals10(os.path.join(data_root_path, "train"),
                        transform=train_transform,
                        preread=preread)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=workers)


def get_test_loader(data_root_path, batch_size, workers,preread):
    test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = Animals10(os.path.join(data_root_path, "test"),
                        transform=test_transform,
                        preread=preread)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=workers)


def get_dataloader(root, bs, num_workers, preread):
    train_loader = get_train_loader(data_root_path=root, batch_size=bs, workers=num_workers,preread=preread)
    test_loader = get_test_loader(data_root_path=root, batch_size=bs, workers=num_workers,preread=preread)
    return train_loader, test_loader


if __name__ == "__main__":
    from visual.tensor_visual import tensor_visual

    train_loader, test_loader = get_dataloader(root=r"D:\WorkSpace\Animals10\dataset", bs=9, num_workers=0, preread=True)
    print(train_loader, test_loader)
    t0 = time.time()
    for input, target in train_loader:
        input.shape
        # tensor_visual(input,nrow=3)
    print(time.time() - t0)
