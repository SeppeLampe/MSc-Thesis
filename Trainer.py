import datetime
import os

import numpy as np
import pytorch_lightning as pl
import torch
from CNN import CNN
from Classifier import *
from DataLoader import *
from pytorch_lightning.loggers import TensorBoardLogger
from skimage.measure import block_reduce
from torch import nn
from torchvision import transforms

LOG_LOCATION = r"/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Logs"
SAVE_LOCATION = r"/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Models"

shards = "/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/OT_Probability_shards/"
shard_bucket = [shards + filename for filename in os.listdir(shards)]

INPUT_TENSOR = torch.load(
    f"/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/Products/OT_Probability_{TEXT}.pt")
LABEL_TENSOR = torch.load(
    "/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/Products/OT_Probability_night.pt")

BATCH_SIZE = 64
NUM_WORKERS = 3
GPUS = 1
EPOCHS = 20
POOL = 5

INPUT_FIELD = 'ot_probability'
TARGET_FIELDS = 'ot_probability'

DAY_MASK = torch.load(
    "/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Dataset/Products/Day_Mask.pt").bool()
DAY_MASK_SIZE = int(torch.sum(torch.Tensor(block_reduce(DAY_MASK, (2 ** POOL, 2 ** POOL), np.max))))

LAKE_MASK = torch.load(r'/theia/data/brussel/vo/000/bvo00012/vsc10262/Thesis_Seppe_Lampe/Lake_Victoria_Mask.pt').bool()
LAKE_MASK_SIZE = int(torch.sum(LAKE_MASK))

CNN_PARAMS = {'input_size': [DAY_MASK_SIZE], 'output_size': 1, 'activation': nn.Sigmoid,
              'size_2d': [], 'kernel_2d': [], 'stride_2d': [], 'pad_2d': [], 'drop_2d': [], 'groups_2d': [],
              'dilation_2d': [],
              'pool_2d': [], 'pool_2d_size': [], 'pool_2d_stride': [], 'pool_2d_pad': [], 'batchnorm_2d': [],
              'size_1d': [36, 36, 18], 'batchnorm_1d': True, 'drop_1d': 0.2, 'final_activation': nn.Sigmoid}

input_transforms = transforms.Compose([
    Reconstruct2D(mask=DAY_MASK),
    Pool(pooled=POOL),
    Mask(mask=DAY_MASK, pooled=POOL)
])

label_transforms = transforms.Compose([
    ToExtreme()
])

if __name__ == '__main__':
    dataset = TensorDataset(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                            input_tensor=INPUT_TENSOR, label_tensor=LABEL_TENSOR,
                            input_transforms=input_transforms, label_transforms=label_transforms, collate_fn=None)
    net = CNN(**CNN_PARAMS)
    classifier = Classifier(model=net, optimizer=torch.optim.Adam, loss_function=my_loss)
    logger = TensorBoardLogger(save_dir=LOG_LOCATION, default_hp_metric=False)
    # Create a trainer which will use 1 GPU, which will run for X epochs and we'll allow it to find the best learning rate to start with
    trainer = pl.Trainer(gpus=GPUS, logger=logger, max_epochs=EPOCHS, progress_bar_refresh_rate=0, auto_lr_find=True,
                         checkpoint_callback=False)  # checkpoint_callback enables/disables checpoint saving every epoch.

    # Train the model with the dataset
    trainer.fit(classifier, dataset)
    # Test the model
    test_params = trainer.test(classifier)
    date_time = datetime.datetime.now().strftime('%d-%m-%Y_%H%M')
    name = SAVE_LOCATION + "/" + date_time + '.pt'
    torch.save(classifier, name)
