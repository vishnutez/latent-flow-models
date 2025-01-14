import torch
import torch.nn as nn

from guided_diffusion.measurements import LinearOperator


class ConditionalVF(nn.Module):
    def __init__(self, model, y, operator: LinearOperator):
        super().__init__()
        self.model = model
        self.y = y
        self.operator = operator

    def b(self, t):
        return 1
    
    def a(self, t):
        return 1

    def forward(self, t, x):

        unconditional_vf = self.model(t, x)
        pred_x1 = 1/self.b(t) * (self.model(t, x)-self.a(t)*x)

        measurement_error = torch.linalg.norm(self.y - self.operator.forward(pred_x1))**2
        vf_correction = torch.autograd.grad(measurement_error, x)[0]

        conditional_vf = unconditional_vf + vf_correction
        
        return conditional_vf