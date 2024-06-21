import torch
from torch import nn
from data.dataset import ImageDataset
from models.UNet_provided import UNet
from utils import train, accuracy_fn, patch_accuracy_fn

data_dir = '/Users/sebastian/University/Master/second_term/cil/road-segmentation/data/training'

device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ImageDataset(data_dir, True, device, use_patches=False, resize_to=(384, 384))
val_dataset = ImageDataset(data_dir, False, device, use_patches=False, resize_to=(384, 384))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

model = UNet().to(device)
loss_fn = nn.BCELoss()
metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
optimizer = torch.optim.Adam(model.parameters())
n_epochs = 5

train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs)