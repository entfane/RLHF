import torch

def rm_loss(good, bad):
    output = -torch.log(torch.sigmoid(good - bad))
    output = output.mean()
    return output