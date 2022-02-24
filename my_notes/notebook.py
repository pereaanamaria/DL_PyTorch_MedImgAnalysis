from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from celluloid import Camera

from dataset_lung import LungDataset
from model import UNet

seq = iaa.Sequential([
    iaa.Affine(translate_percent=(0.15), scale=(0.85, 1.15), rotate=(-45,45)),
    iaa.ElasticTransformation()
])

train_path = Path('Data/Atrium/Task06_Lung/Preprocessed/train/')
val_path = Path('Data/Atrium/Task06_Lung/Preprocessed/val/')

train_dataset = LungDataset(train_path, seq)
val_dataset = LungDataset(val_path, None)

# target_list = []
# print(f'Start oversampling')
#
# for _, label in train_dataset:
#     if np.any(label):
#         target_list.append(1)
#     else:
#         target_list.append(0)
#
# unique = np.unique(target_list, return_counts=True)
# fraction = unique[1][0] / unique[1][1]
#
# print(f'Stop oversampling :: fraction = {fraction} ')
#
# weight_list = []
#
# for target in target_list:
#     if target == 0:
#         weight_list.append(1)
#     else:
#         weight_list.append(fraction)
#
# sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1,
#                                            num_workers=4, sampler=sampler)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1,
                                           num_workers=4, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1,
                                         num_workers=4, shuffle=False)


class LungSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = UNet()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()

        pred = self(ct.float())
        loss = self.loss_fn(pred, mask)

        self.log('Train Dice', loss)
        if batch_idx % 50 == 0:
            self.log_images(ct.cpu(), pred.cpu(), mask.cpu(), 'Train')

        return loss

    def validation_step(self, batch, batch_idx):
        ct, mask = batch
        mask = mask.float()

        pred = self(ct.float())
        loss = self.loss_fn(pred, mask)

        self.log('Val Dice', loss)
        if batch_idx % 50 == 0:
            self.log_images(ct.cpu(), pred.cpu(), mask.cpu(), 'Val')

        return loss

    def log_images(self, ct, pred, mask, name):
        pred = pred > 0.5

        fig, axis = plt.subplots(1, 2)

        axis[0].imshow(ct[0][0], cmap='bone')
        mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")

        axis[1].imshow(ct[0][0], cmap='bone')
        mask_ = np.ma.masked_where(pred[0][0] == 0, pred[0][0])
        axis[1].imshow(mask_, alpha=0.6, cmap='autumn')
        axis[1].set_title("Pred")

        self.logger.experiment.add_figure(name, fig, self.global_step)

    def configure_optimizers(self):
        return [self.optimizer]


model = LungSegmentation()
checkpoint_callback = ModelCheckpoint(monitor='Val Dice', save_top_k=30, mode='min')

trainer = pl.Trainer(gpus=1, logger=TensorBoardLogger(save_dir='logs/lungs'), log_every_n_steps=1,
                     callbacks=checkpoint_callback, max_epochs=30)
trainer.fit(model, train_loader, val_loader)

