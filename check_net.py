import torch
import torch.nn as nn

model_path = "/home/lorenzo/Github/University/IsaacLab/logs/rsl_rl/quadcopter_direct/2025-02-16_01-28-44/model_3450.pt"

# Caricare solo lo state_dict
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Stampare i nomi dei layer
for layer_name in state_dict.keys():
    print(layer_name)
