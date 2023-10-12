import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import math

import matplotlib.pyplot as plt
 
X = torch.tensor([[-1,-1,-1], [-1,1,-1], [1,-1,-1], [1,1,-1],
                  [-1,-1,1], [-1,1,1], [1,-1,1], [1,1,1]])

def dac (binary_value) :
  factor = 1.0
  total = 0.0
  for bit in binary_value :
    total = total + bit * factor
    factor = factor * 2.0

  return total

ADC_RANGE = 1.0
def adc (real_value) :
  n = ADC_RANGE
  x = real_value
  bits = []
  for i in range(8) :
    n = 0.5 * n
    diff = x - n
    bit = torch.nn.functional.sigmoid(diff)
    remainder = bit * diff
    x = x - remainder
    bits.append(bit)
  return bits



def mux (s, a, b) :
  return b if s > 0.5 else a

def dataset (x) :

  a = 0 if x[0] < -0.5 else 1
  b = 0 if x[1] < -0.5 else 1
  c = 0 if x[2] < -0.5 else 1

  # sum = (a ^ b) ^ c

  tmp = mux(b, 1, 0)

  sum = mux(c,
    mux(a, b, tmp),
    mux(a, tmp, 0))

  ret_val = -1 if sum < 0.5 else 1
  print(a, b, c, sum, ret_val)
  return ret_val
  
  '''
  # return 1.0 if x[0] <= -10 else 0.5 if x[0] < 10 else 0.0
  if x[0] < -0.5 and x[1] < -0.5 :
    return -1
  elif x[0] > 0.5 and x[1] < -0.5 :
    return 1
  elif x[0] < -0.5 and x[1] > 0.5 :
    return 1
  elif x[0] > 0.5 and x[1] > 0.5 :
    return -1
  '''

Y = torch.tensor([dataset(x) for x in X])

#plt.plot(X, Y)
#plt.show()

def ReLU (x) :
  return x if x > 0 else 0

def Mux (s, a, b) :
  return 0.5 * (s * a - s * b + a + b)

def Cmos (x) :
  return 4.0 * torch.nn.functional.sigmoid(x) - 2.0

print('Test MUX')
print(Mux(-1, -1, -1))
print(Mux(-1, 1, -1))
print(Mux(-1, -1, 1))
print(Mux(-1, 1, 1))
print(Mux(1, -1, -1))
print(Mux(1, 1, -1))
print(Mux(1, -1, 1))
print(Mux(1, 1, 1))

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

        self._dL_by_dw0 = 0.0
        self._dL_by_dw1 = 0.0

        L = 4
        N = 0.02

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
        torch.nn.init.normal_(self.l1.weight, mean=0, std=0.02)
        #torch.nn.init.constant_(self.l1.weight[0][0], 1.0)
        #torch.nn.init.constant_(self.l1.weight[1][0], -1.0)

        #torch.nn.init.normal_(self.l1.weight, mean=0, std=N/math.sqrt(L))
        torch.nn.init.normal_(self.l2.weight, mean=0, std=N/math.sqrt(L))
        torch.nn.init.normal_(self.l3.weight, mean=0, std=N/math.sqrt(L))
        torch.nn.init.normal_(self.l4.weight, mean=0, std=N/math.sqrt(L))

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
      '''
      m00 = Mux(x[0], self.l1.weight[0][0], self.l1.weight[1][0])
      m01 = Mux(x[0], self.l1.weight[2][0], self.l1.weight[3][0])
      m10 = Mux(x[1], m00, m01)
      '''

      a = x[0]
      b = x[1]
      c = x[2]

      # m00 = Mux(b, self.l1.weight[0][0], self.l1.weight[1][0]) # self.l1.weight[0][0], -1.0) #self.l1.weight[1][0])

      m00 = Mux(b, self.l1.weight[0][0], self.l1.weight[1][0])

      # incoming backprop(2 users):
      #  from mb: 0.5 * (a + 1.0) * 0.5 * (1.0 - c) * dloss_by_dmodel
      #  from ma: 0.5 * (1.0 - a) * 0.5 * (c + 1.0) * dloss_by_dmodel

      # backprop to w0: 0.5 * (b + 1.0) * (0.5 * (a + 1.0) * 0.5 * (1.0 - c) * dloss_by_dmodel + 0.5 * (1.0 - a) * 0.5 * (c + 1.0) * dloss_by_dmodel)

      # backprop to w1: 0.5 * (1.0 - b) * (0.5 * (a + 1.0) * 0.5 * (1.0 - c) * dloss_by_dmodel + 0.5 * (1.0 - a) * 0.5 * (c + 1.0) * dloss_by_dmodel)
      
      
      # don't backpropagate to the training data inputs
      # dm00_by_b = 0.5 * (self.l1.weight[0][0] - self.l1.weight[1][0])
      dm00_by_w0 = 0.5 * (b + 1.0)
      dm00_by_w1 = 0.5 * (1.0 - b)

      # Old code: m10 = 0.000001 * self.l1.bias[0] + Mux(c, ...


      ma = Mux(a, self.l1.weight[0][0], m00)
      
      # incoming backprop (1 user): 0.5 * (c + 1.0) * dloss_by_dmodel

      # backprop to w0: 0.5 * (a + 1.0) * 0.5 * (c + 1.0) * dloss_by_dmodel
      # backprop to m00: 0.5 * (1.0 - a) * 0.5 * (c + 1.0) * dloss_by_dmodel

      # dma_by_a = 0.5 * (self.l1.weight[0][0] - m00)
      dma_by_w0 = 0.5 * (a + 1.0)
      dma_by_m00 = 0.5 * (1.0 - a)


      mb = Mux(a, m00, b)
      
      # incoming backprop (1 user): 0.5 * (1.0 - c) * dloss_by_dmodel

      # backprop to m00: 0.5 * (a + 1.0) * 0.5 * (1.0 - c) * dloss_by_dmodel

      # dmb_by_a = 0.5 * (m00 - b)
      dmb_by_m00 = 0.5 * (a + 1.0)
      #dmb_by_b = 0.5 * (1.0 - a)

      # backprop: to ma: 0.5 * (c + 1.0)    to mb: 0.5 * (1.0 - c)

      m10 = Mux(c, ma, mb)

      # Mux(s, a, b) = 0.5 * (s * a - s * b + a + b)
      # dm10/ds = 0.5 * (a - b)
      # dm10/da = 0.5 * (s + 1.0)
      # dm10/db = 0.5 * (1.0 - s)
      # dm10_by_dc = 0.5 * (ma - mb)
      dm10_by_dma = 0.5 * (c + 1.0) #* dloss_by_dmodel
      dm10_by_dmb = 0.5 * (1.0 - c) #* dloss_by_dmodel

      # print(m00, b, a, c, m10)
      

      return m10

      # return torch.nn.functional.sigmoid(Mux(x, self.l1.weight[0][0], self.l1.weight[1][0]))

      # return torch.nn.functional.silu(self.l4(torch.nn.functional.silu(self.norm3(self.l3(torch.nn.functional.silu(self.norm2(self.l2(torch.nn.functional.silu(self.norm1(self.l1(x)))))))))))
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

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)

# The optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0, momentum=0.0)

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.1)

scheduler = MultiStepLR(optimizer, milestones=[100,300,500,800], gamma=0.5)

# Define the training loop
epochs = 120
cost = []
total = 0

dl_by_dm = list(range(len(X)))
dl_by_dw0 = list(range(len(X)))
dl_by_dw1 = list(range(len(X)))

for epoch in range(epochs):
    total = 0
    epoch = epoch + 1

    network = 0.0

    # for x, y in zip(X, Y) :
    for i in range(len(X)) :
      # x, y in zip(X, Y) :
        x = X[i] + 0.01 * np.random.normal()
        y = Y[i] #  + 0.00001 * np.random.normal()

        # forward
        yhat = model(x)

        # backprop value of dL/dW
        # dl0 / dmodel_0 = 2 * (m0 - y0)
        dl_by_dm[i] = 2 * (yhat - y)
        dloss_by_dmodel = 2 * (yhat - y)
        a = x[0]
        b = x[1]
        c = x[2]
        dl_by_dw0[i] = 0.5 * (b + 1.0) * (0.5 * (a + 1.0) * 0.5 * (1.0 - c) * dloss_by_dmodel
                                        + 0.5 * (1.0 - a) * 0.5 * (c + 1.0) * dloss_by_dmodel) + 0.5 * (a + 1.0) * 0.5 * (c + 1.0) * dloss_by_dmodel

        dl_by_dw1[i] = 0.5 * (1.0 - b) * (0.5 * (a + 1.0) * 0.5 * (1.0 - c) * dloss_by_dmodel
                                        + 0.5 * (1.0 - a) * 0.5 * (c + 1.0) * dloss_by_dmodel)

        network = network + torch.square(torch.sub(y, yhat))

    # backward

    # total loss is network, made of multiple networks fed with different x & y:
    # Sum_i( (model(w1, w2 | x_i) - y_i) ^ 2 ):
    # l0 = (model_0 - y_0) ^ 2 = m0^2 - 2*y0 * m0 + y0^2 | dl0 / dmodel_0 = (2 * m0 - 2*y0)
    # l1 = (model_1 - y_1) ^ 2
    # Total_Loss = (l0 + l1 + l2 + ...) -> dTotal/dl0 = 1
    dLtotal_by_dw0 = 0.0
    dLtotal_by_dw1 = 0.0
    for i in range(len(X)) :
      dLtotal_by_dw0 = dLtotal_by_dw0 + dl_by_dw0[i]
      dLtotal_by_dw1 = dLtotal_by_dw1 + dl_by_dw1[i]

    new_w0_brians = model.l1.weight[0][0]-optimizer.param_groups[0]["lr"] * (dLtotal_by_dw0)
    new_w1_brians = model.l1.weight[1][0]-optimizer.param_groups[0]["lr"] * (dLtotal_by_dw1)

    #print('new w0 =', model.l1.weight[0][0]-optimizer.param_groups[0]["lr"] * (dLtotal_by_dw0))
    #print('new w1 =', model.l1.weight[1][0]-optimizer.param_groups[0]["lr"] * (dLtotal_by_dw1))

    #print('Weights before update:', 
    #  model.l1.weight[0][0],
    #  model.l1.weight[1][0])
    network.backward()

    lr1 = optimizer.param_groups[0]["lr"]
    print(epoch, network, lr1)

    ''' 
    with torch.no_grad():
      if model.l1.weight[1][0] < 0.1 and model.l1.weight[1][0] > -0.1 :
        torch.nn.init.normal_(model.l1.weight[1][0], mean=0, std=2.0)
      elif model.l1.weight[1][0] > 1.0 :
        torch.nn.init.normal_(model.l1.weight[1][0], mean=0.98, std=0.02)
      elif model.l1.weight[1][0] < -1.0 :
        torch.nn.init.normal_(model.l1.weight[1][0], mean=-0.98, std=0.02)

      if model.l1.weight[0][0] < 0.1 and model.l1.weight[0][0] > -0.1 :
        torch.nn.init.normal_(model.l1.weight[0][0], mean=0, std=2.0)
      elif model.l1.weight[0][0] > 1.0 :
        torch.nn.init.normal_(model.l1.weight[0][0], mean=0.98, std=0.02)
      elif model.l1.weight[0][0] < -1.0 :
        torch.nn.init.normal_(model.l1.weight[0][0], mean=-0.98, std=0.02)

      if model.l1.weight[2][0] < 0.005 and model.l1.weight[2][0] > -0.005 :
        torch.nn.init.normal_(model.l1.weight[2][0], mean=0, std=1.0)
      elif model.l1.weight[2][0] > 0.1 :
        torch.nn.init.normal_(model.l1.weight[2][0], mean=1.0, std=0.01)
      elif model.l1.weight[2][0] < -0.1 :
        torch.nn.init.normal_(model.l1.weight[2][0], mean=-1.0, std=0.01)

      if model.l1.weight[3][0] > 1.0 :
        torch.nn.init.normal_(model.l1.weight[3][0], mean=1.0, std=0.01)
      elif model.l1.weight[3][0] < -1.0 :
        torch.nn.init.normal_(model.l1.weight[3][0], mean=-1.0, std=0.01)
    '''

    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.01)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    print('Weight difference after update:', 
      model.l1.weight[0][0] - new_w0_brians,
      model.l1.weight[1][0] - new_w1_brians)

    '''
    print('Gradients:',
      model.l1.weight.grad[0][0],
      model.l1.weight.grad[1][0],
      model._dL_by_dw0,
      model._dL_by_dw1
      )
    '''

    optimizer.zero_grad()
    # get total loss 
    total += network.item() 

    cost.append(total)

print(model.l1.state_dict())
# print(model.l2.state_dict())

# plot the result of function approximator
#plt.plot(X.numpy(), model(X).detach().numpy())
#plt.plot(X.numpy(), Y.numpy(), 'm')
#plt.xlabel('x')
#plt.show()
 
# plot the cost
plt.plot(cost)
plt.xlabel('epochs')
plt.title('cross entropy loss')
plt.show()

for i in range(len(X)) :
  x = X[i]
  y_hat = model(x)
  print(x[0], x[1], x[2], 'y_hat y_should', y_hat, Y[i])

