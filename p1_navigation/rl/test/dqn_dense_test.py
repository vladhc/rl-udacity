import torch
from torch.autograd import Variable
from torchtest import assert_vars_change
import torch.nn.functional as F

import rl


state_size = 4
action_size = 2
batch_size = 20

states = Variable(torch.randn(batch_size, state_size))
q_values = Variable(torch.randn(batch_size, action_size))
batch = [states, q_values]
model = rl.DQNDense(state_size, action_size).to('cuda')

assert_vars_change(
    model=model,
    loss_fn=F.smooth_l1_loss,
    optim=torch.optim.Adam(model.parameters()),
    batch=batch)
