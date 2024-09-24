import torch
import torch.nn.functional as F
from model import FlowNetS

"""
def get_gradient(input, dim):
    
    assert isinstance(input, torch.Tensor)
    batch, channel, row, col = input.shape
    
    if dim == 'x':
        filter = torch.tensor([-1, 0, 1], dtype = torch.long).view(1, 1, 1, 3).to(input.device)
        gradient_x = F.conv2d(input, filter, padding = (0, 1))
        return gradient_x
    
    elif dim == 'y':
        filter = torch.tensor([-1, 0, 1], dtype = torch.long).view(1, 1, 3, 1).to(input.device)
        gradient_y = F.conv2d(input, filter, padding = (1, 0))
        return gradient_y

# image_data = torch.randn(1, 1, 5, 5)
image_data = torch.randint(low = 1, high = 10, size = (1, 1, 5, 5))
print(image_data)

result = get_gradient(image_data, 'y')
print(result)
"""


model = FlowNetS()
a = torch.randn(1, 6, 512, 512)
layer_outputs = {}

with torch.no_grad():
    
    layer_outputs = {'inputs' : a}
    
    for name, layer in model.named_children():
        a = layer(a)
        layer_outputs[name] = a