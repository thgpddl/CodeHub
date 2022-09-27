import os.path

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_root, mode="gpu"):
        super(TrainPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=4)
        self.rn = ops.random.CoinFlip()

        self.input = ops.readers.File(file_root=data_root, random_shuffle=False)
        self.decode = ops.decoders.Image(device='mixed')  # 解码
        self.resize = ops.Resize(resize_x=255, resize_y=255, device=mode)
        self.colortwist = ops.ColorTwist(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, device=mode)
        self.rotate = ops.Rotate(angle=5, device=mode)
        self.randomresizedcrop = ops.RandomResizedCrop(size=[224, 224], device=mode)
        self.cropmirrornormalize = ops.CropMirrorNormalize(mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                                           std=[0.5 * 255, 0.5 * 255, 0.5 * 255], device=mode)

    # 作用是在调用该pipeline时，应该如何对数据进行实际的操作，可以理解为pytorch的module的forward函数
    def define_graph(self):
        jpegs, labels = self.input(name='Reader')  # readers
        images = self.decode(jpegs)  # decoders
        images = self.resize(images)  # Resize
        images = self.colortwist(images)  # Resize
        images = self.rotate(images)  # Resize
        images = self.randomresizedcrop(images)  # Resize
        images = self.cropmirrornormalize(images, mirror=self.rn(probability=0.5))  # Resize
        return images, labels


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_root, mode="gpu"):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=4)

        self.input = ops.readers.File(file_root=data_root, random_shuffle=False)
        self.decode = ops.decoders.Image(device='mixed')  # 解码
        self.resize = ops.Resize(resize_x=224, resize_y=224, device=mode)
        self.cropmirrornormalize = ops.CropMirrorNormalize(mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                                           std=[0.5 * 255, 0.5 * 255, 0.5 * 255], device=mode)

    # 作用是在调用该pipeline时，应该如何对数据进行实际的操作，可以理解为pytorch的module的forward函数
    def define_graph(self):
        jpegs, labels = self.input(name='Reader')  # readers
        images = self.decode(jpegs)  # decoders
        images = self.resize(images)  # Resize
        images = self.cropmirrornormalize(images)  # Normalize
        return images, labels


def get_dataloader(bs, num_workers, root, device_id=0):
    pipe_train = TrainPipeline(bs, num_workers, device_id, os.path.join(root, 'train'))
    pipe_train.build()
    train_loader = DALIClassificationIterator(pipe_train, size=pipe_train.epoch_size('Reader'),
                                              last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    pipe_test = TestPipeline(bs, num_workers, device_id, os.path.join(root, 'test'))
    pipe_test.build()
    test_loader = DALIClassificationIterator(pipe_test, size=pipe_test.epoch_size('Reader'),
                                             last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    return train_loader, test_loader


if __name__ == "__main__":
    import time

    train_loader, test_loader = get_dataloader(bs=64,
                                               num_workers=4,
                                               root="/home/ubuntu/WorkSpace/Animals10/dataset/")
    t0=time.time()
    for data in train_loader:
        image = data[0]['data']
        targe = data[0]['label']
        target = targe.view(-1).long().to("cuda:0")
    # for data in test_loader:
    #     image = data[0]['data']
    #     targe = data[0]['label']
    # tensor_visual(image.cpu(),nrow=8)
    print(time.time() - t0)  # 3.155s
