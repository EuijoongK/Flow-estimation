import torch
import torch.nn as nn
import torch.nn.functional as F

def get_gradient(input, dim):
    
    assert isinstance(input, torch.Tensor)
    batch, channel, row, col = input.shape
    
    if dim == 'x':
        filter = torch.tensor([-1, 0, 1], dtype = torch.float32).view(1, 1, 1, 3).to(input.device)
        gradient_x = F.conv2d(input, filter, padding = (0, 1))
        return gradient_x
    
    elif dim == 'y':
        filter = torch.tensor([-1, 0, 1], dtype = torch.float32).view(1, 1, 3, 1).to(input.device)
        gradient_y = F.conv2d(input, filter, padding = (1, 0))
        return gradient_y

class CustomFlowLoss(nn.Module):
    
    def __init__(self, epsilon = 2e-7):
        super(CustomFlowLoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, prev_frame, current_frame, u, v):
        
        """
        assert isinstance(prev_frame, torch.Tensor)
        assert isinstance(current_frame, torch.Tensor)
        assert isinstance(prediction, torch.Tensor)
        
        x_diff = torch.gradient(prev_frame, dim = 1)
        y_diff = torch.gradient(prev_frame, dim = 0)
        """
        
        x_diff = get_gradient(current_frame, 'x')
        y_diff = get_gradient(current_frame, 'y')            
        t_diff = current_frame - prev_frame
        
        OF_loss = torch.pow(torch.mul(u, x_diff) + torch.mul(v, y_diff) + t_diff, 2) + self.epsilon
        sum_error = OF_loss.sum(dim = (2, 3))
 
        return torch.mean(sum_error)
        
        
'''

loss_fn = CustomLoss()
loss = loss_fn(predictions, targets)

'''