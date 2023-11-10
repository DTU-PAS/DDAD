import torch
import os
import torch.nn as nn
from forward_process import *
from dataset import *

from dataset import *
from test import *
from loss import *
from sample import *
from tqdm import tqdm
from torchmetrics.aggregation import MeanMetric

def trainer(model, category, config):
    '''
    Training the UNet model
    :param model: the UNet model
    :param category: the category of the dataset
    '''
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.model.learning_rate, weight_decay=config.model.weight_decay
    )
    if config.data.name == "PBA":
        train_dataset = PhenoBenchAnomalyDataset_DDAD(config.data.data_dir, "train", 0.0, config, overfit=config.model.overfit)
    else:
        train_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=category,
            config = config,
            is_train=True,
        )
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.model.num_workers,
        drop_last=True,
    )
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists(config.model.checkpoint_dir):
        os.mkdir(config.model.checkpoint_dir)

    with tqdm(range(config.model.epochs)) as t_epoch:
        for epoch in t_epoch:
            mean_loss = MeanMetric()
            for step, batch in enumerate(trainloader):
                if (step * config.data.batch_size) > 500:
                    break
                t = torch.randint(0, config.model.trajectory_steps, (batch[0].shape[0],), device=config.model.device).long()
                optimizer.zero_grad()
                loss = get_loss(model, batch[0], t, config) 
                loss.backward()
                optimizer.step()
                mean_loss.update(loss.item())
                # t_epoch.set_description(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {mean_loss.compute()}")
                # t_epoch.set_description(f"Epoch {epoch} | Loss: {loss.item()}")
            if epoch % 250 == 0:
                if config.model.save_model:
                    model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
                    if not os.path.exists(model_save_dir):
                        os.mkdir(model_save_dir)
                    torch.save(model.state_dict(), os.path.join(model_save_dir, str(epoch)))
                    
    if config.model.save_model:
        model_save_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)
        torch.save(model.state_dict(), os.path.join(model_save_dir, str(config.model.epochs)))