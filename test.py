import torch
print(torch.__version__)  # Check your PyTorch version
print(torch.cuda.is_available())

import torch
x = torch.rand(5, 5).cuda()
print(x)
