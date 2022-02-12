from network import FCCA
import torch

policy_model = FCCA(100, 128, [-10, 10])
traced_cell = torch.jit.script(policy_model)
traced_cell.save("./policy_init.pt")