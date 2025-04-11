from share import *
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint

# Configs
resume_path = '/root/autodl-tmp/checkpoints/epoch=399-step=36399.ckpt'
batch_size = 16
logger_freq = 600
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()

target_dict = {}
checkpoint = torch.load(resume_path, map_location='cpu')
pretrained_weights = checkpoint['state_dict']
for k in pretrained_weights.keys():
    target_dict[k] = pretrained_weights[k].clone()
model.load_state_dict(target_dict,strict=False)


model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

checkpoint_callback = ModelCheckpoint(
    dirpath='/root/autodl-tmp/checkpoints',
    save_top_k = 1,
    monitor='epoch',
    mode='max'
)

# # Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
                    gpus=1, 
                    precision=32, 
                    max_epochs = 200, 
                    callbacks=[checkpoint_callback,logger], 
                    # resume_from_checkpoint='/root/autodl-tmp/checkpoints/epoch=199-step=18199.ckpt',
                    )



# Train!
trainer.fit(model, dataloader)
