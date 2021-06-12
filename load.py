import os
import torch

# import settings
# from model import Model
from utils.optimizer import get_optimizer

def load_checkpoint(filepath, model, device, **params):

        model = model.to(device)
        
        if os.path.exists(filepath):
                checkpoint = torch.load(filepath)
                model.load_state_dict(checkpoint['model_stat'])
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
                # optimizer = get_optimizer(model.parameters(), params["optimizer"])
                optimizer.load_state_dict(checkpoint['optimizer_stat'])
                print("Use pretrain weight.")
        else:
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
                print("Initialize optimizer.")
        return model, optimizer