import os
import torch

# folder in same project
base_dir = os.path.dirname(__file__)         # folder of this script
path = os.path.join(base_dir, "arousal_head.pt")

state = torch.load(path, map_location="cpu")

print(state.keys())
print("\nModel state_dict keys:")
print(state["state_dict"].keys())

print("\nMetadata:")
for k, v in state.items():
    if k != "state_dict":
        print(f"{k}: {v}")
