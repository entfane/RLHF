import torch.nn as nn
import torch

class RMLoss(nn.Module):

    def forward(self, good, bad):
        """
        Runs the forward pass.
        """
        output = -torch.log(torch.sigmoid(good - bad))
        output = output.mean()
        
        return output

    