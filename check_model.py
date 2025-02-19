import torch

model_path = "/home/lorenzo/Github/University/IsaacLab/logs/rsl_rl/quadcopter_direct/2025-02-16_01-28-44/model_3450.pt"
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = checkpoint["model_state_dict"]

# Trova la dimensione dell'output
actor_output_size = state_dict['actor.4.weight'].shape[0]
critic_output_size = state_dict['critic.4.weight'].shape[0]

print(f"Output della rete Actor: {actor_output_size}")
print(f"Output della rete Critic: {critic_output_size}")
