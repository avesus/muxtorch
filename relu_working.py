import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import math

import matplotlib.pyplot as plt
 
# generate synthetic the data
X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] <= -10)] = 1.0
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0
 
#plt.plot(X, Y)
#plt.show()

def ReLU (x) :
  return x if x > 0 else 0

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class one_layer_net (torch.nn.Module):    

    def __init__ (self):
        super(one_layer_net, self).__init__()

        L = 15
        N = 0.02
        B = 0.02
        zero_bias = True

        self.l1 = torch.nn.Linear(1, L)
        self.l2 = torch.nn.Linear(L, L)
        self.l3 = torch.nn.Linear(L, L)
        self.l4 = torch.nn.Linear(L, 1)
        self.norm1 = RMSNorm(L)
        self.norm2 = RMSNorm(L)
        self.norm3 = RMSNorm(L)

        ###
        ### Analog of torch.nn.init.xavier_uniform_(self.l1.weight)
        ###
        torch.nn.init.normal_(self.l1.weight, mean=0, std=N/math.sqrt(L))
        torch.nn.init.normal_(self.l2.weight, mean=0, std=N/math.sqrt(L))
        torch.nn.init.normal_(self.l3.weight, mean=0, std=N/math.sqrt(L))
        torch.nn.init.normal_(self.l4.weight, mean=0, std=N/math.sqrt(L))

        if not zero_bias :
          torch.nn.init.normal_(self.l1.bias, mean=0, std=B/math.sqrt(L))
          torch.nn.init.normal_(self.l2.bias, mean=0, std=B/math.sqrt(L))
          torch.nn.init.normal_(self.l3.bias, mean=0, std=B/math.sqrt(L))
          torch.nn.init.normal_(self.l4.bias, mean=0, std=B)
        else :
          torch.nn.init.constant_(self.l1.bias, 0.0)
          torch.nn.init.constant_(self.l2.bias, 0.0)
          torch.nn.init.constant_(self.l3.bias, 0.0)
          torch.nn.init.constant_(self.l4.bias, 0.0)

        print('w00, w01', self.l1.weight[0][0], self.l1.weight[1][0])
        print('b00, b01', self.l1.bias[0], self.l1.bias[1])

        print('w10, w11', self.l2.weight[0][0], self.l2.weight[0][1])
        print('b10', self.l2.bias[0])

        print(self.l1.state_dict())
        print(self.l2.state_dict())

    def forward (self, x) :
      return torch.nn.functional.silu(self.l4(torch.nn.functional.silu(self.norm3(self.l3(torch.nn.functional.silu(self.norm2(self.l2(torch.nn.functional.silu(self.norm1(self.l1(x)))))))))))
      '''
        torch_tensors = (
          torch.nn.functional.relu(
          self.l2(
            #torch.nn.functional.relu(
              self.norm(
                        self.l1(x)
              )
            #)
          )
         )
        )

        return torch_tensors
        '''
 
# create the model 
model = one_layer_net()
 

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.002)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)

scheduler = MultiStepLR(optimizer, milestones=[100,300,500,800], gamma=0.5)

# Define the training loop
epochs = 1000
cost = []
total = 0


for epoch in range(epochs):
    total = 0
    epoch = epoch + 1

    network = 0.0

    for x, y in zip(X, Y) :

        # forward
        yhat = model(x)

        network = network + torch.square(torch.sub(y, yhat))

    # backward
    network.backward()

    lr1 = optimizer.param_groups[0]["lr"]
    print(epoch, network, lr1)


    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.01)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    # get total loss 
    total += network.item() 

    cost.append(total)

print(model.l1.state_dict())
print(model.l2.state_dict())

# plot the result of function approximator
plt.plot(X.numpy(), model(X).detach().numpy())
plt.plot(X.numpy(), Y.numpy(), 'm')
plt.xlabel('x')
plt.show()
 
# plot the cost
plt.plot(cost)
plt.xlabel('epochs')
plt.title('cross entropy loss')
plt.show()

