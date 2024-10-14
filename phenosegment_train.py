from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from phenobench_dataset import Phenobench
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = '/netscratch/naeem/controlnet/controlnet-sd14-ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = Phenobench(root_dir='/netscratch/naeem/phenobench')
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, 
        precision=32, 
        callbacks=[logger], 
        default_root_dir="/netscratch/naeem/controlnet/phenobench_train/",
        max_epochs=100)

# Train!
trainer.fit(model, dataloader)
