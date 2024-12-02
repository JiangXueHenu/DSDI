import torch
import torch.nn as nn

class arModel(nn.Module):
    def __init__(self, input_dim, ar_order=1):
        super(arModel, self).__init__()
        self.ar_order = ar_order
        self.fc = nn.Linear(input_dim * ar_order, input_dim)

    def forward(self, x):
        B, K, L = x.shape
        output = []

        for t in range(self.ar_order, L):
            x_ar = x[:, :, t - self.ar_order:t]
            x_ar = x_ar.view(B, K * self.ar_order)
            output_step = self.fc(x_ar)
            output.append(output_step.unsqueeze(2))

        output = torch.cat(output, dim=2)


        padding = torch.zeros(B, K, self.ar_order, device=x.device)
        output = torch.cat((padding, output), dim=2)

        return output