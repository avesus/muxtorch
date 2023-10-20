import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import math
import random

# torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt

'''
TODO: Prepare gradually increasing in complexity training data sets.
Try to train all possible LUT-3 and LUT-4 functions first,
using the standard LUT NN.

Also, try to train the simple functions first using non-LUT configurations:
A MUX can be routed from any of the two inputs, or "1", or "0":

Three trainable selector MUXes:
SEL -> MUX_C
SEL -> MUX_A
SEL -> MUX_B

Each "SEL" has two parameters to route from inputs A, B, from 0, or from 1.

Total: 6 parameters. Same 2-variables Boolean functions. Will it learn?
'''
 
X = torch.tensor([[-1,-1], [-1,1], [1,-1], [1,1]])
Y = torch.zeros(X.shape[0])

def nn_to_py (nn) :
  return 1 if 0.5 * (nn + 1.0) > 0.5 else 0

def py_to_nn (py) :
  return py * 2.0 - 1.0

def bit_not (n, numbits=1) :
    return (1 << numbits) - 1 - n

def dataset (x) :
  a = nn_to_py(x[0])
  b = nn_to_py(x[1])

  #return py_to_nn(a | b)
  #print('NXOR', bit_not(a ^ b))
  #return py_to_nn(bit_not(a ^ b))
  return [
    py_to_nn(a ^ b),
    py_to_nn((a & b) ^ 0) ]

  #return py_to_nn(a & b)

Y = torch.tensor([dataset(x) for x in X])

#plt.plot(X, Y)
#plt.show()

def Mux (s, a, b) :
  # return b if s > 0 else a
  val = 0.5 * (s * b - s * a + a + b)
  # print(s, a, b, val)
  return val

'''
print('Test MUX')
print(Mux(-1, -1, -1))
print(Mux(-1, 1, -1))
print(Mux(-1, -1, 1))
print(Mux(-1, 1, 1))
print(Mux(1, -1, -1))
print(Mux(1, 1, -1))
print(Mux(1, -1, 1))
print(Mux(1, 1, 1))
'''

torch_w0_grad = 0.0

def on_w0_grad (grad) :
  #global torch_w0_grad
  print('w0.grad', grad.item())

  #torch_w0_grad += grad.item()
  return grad

class LUT2 (torch.nn.Module) :
  def __init__ (self) :
    super().__init__()

    self.weights = torch.nn.Parameter(torch.zeros(4))
    # TODO: divide by sqrt(2 * num_layers)
    #torch.nn.init.normal_(self.weights, mean=0, std=0.02)
    #torch.nn.init.normal_(self.weights, mean=0, std=2.0)

  def forward (self, a, b, noise) :
    #noise = 1.0
    #noise = 1e-10 #1.0
    noise = 0.125


    #noise = 0.02


    return Mux(b,
      Mux(a,
        self.weights[0] + torch.abs(torch.sub(1.0, torch.abs(self.weights[0]))) * np.random.normal() * noise,
        self.weights[1] + torch.abs(torch.sub(1.0, torch.abs(self.weights[1]))) * np.random.normal() * noise
      ),
      Mux(a,
        self.weights[2] + torch.abs(torch.sub(1.0, torch.abs(self.weights[2]))) * np.random.normal() * noise,
        self.weights[3] + torch.abs(torch.sub(1.0, torch.abs(self.weights[3]))) * np.random.normal() * noise
      ))

class Triangle (torch.nn.Module):    

  def __init__ (self):
    super(Triangle, self).__init__()
    self.luts = torch.nn.ModuleList()
    for lut in range(3):
      self.luts.append(LUT2())
    self.sstate = [torch.tensor(-1), torch.tensor(-1), torch.tensor(-1)]

  def uupdate (self) :
    self.sstate[0] = self.new_state[0]#.detach()
    self.sstate[1] = self.new_state[1]#.detach()
    self.sstate[2] = self.new_state[2]#.detach()

  def forward (self, x0, x1, x2, noise) :
    new_state = [
      self.luts[0].forward(x1, x2, noise),
      self.luts[1].forward(x0, x2, noise),
      self.luts[2].forward(x0, x1, noise)]
    self.new_state = new_state
    return new_state
 

class one_layer_net (torch.nn.Module):    

    def __init__ (self):
        super(one_layer_net, self).__init__()
        self.fabric = torch.nn.ModuleList()
        for i in range(9) :
          self.fabric.append(Triangle())

    def forward (self, x, dummy1, dummy2) :


      # RNN_STEPS = 7 # learns with 0.125 noise
      RNN_STEPS = 130

      prev = []
      for i in range(len(self.fabric)) :
        prev.append([ torch.tensor(-1), torch.tensor(-1), torch.tensor(-1) ])

      a = x[0]
      b = x[1]

      for i in range(RNN_STEPS) :
        self.fabric[0](prev[1][0], prev[0][1], prev[5][2], dummy2)
        self.fabric[1](prev[0][0], prev[2][1], prev[1][2], dummy2)
        self.fabric[2](prev[2][0], prev[1][1], prev[3][2], dummy2)
        self.fabric[3](prev[4][0], prev[6][1], prev[2][2], dummy2)
        self.fabric[4](prev[3][0], prev[5][1], b, dummy2)
        self.fabric[5](prev[5][0], prev[4][1], prev[0][2], dummy2)
        self.fabric[6](prev[6][0], prev[3][1], a, dummy2)
        self.fabric[7](prev[2][0], prev[7][1], prev[7][2], dummy2)
        self.fabric[8](prev[5][0], prev[8][1], prev[8][2], dummy2)

        #if i > 2 :
        #  a = torch.tensor(-1)
        #  b = torch.tensor(-1)

        for j in range(len(self.fabric)) :
          self.fabric[j].uupdate()

        for j in range(len(self.fabric)) :
          prev[j][0] = self.fabric[j].sstate[0]
          prev[j][1] = self.fabric[j].sstate[1]
          prev[j][2] = self.fabric[j].sstate[2]

      # return self.fabric[0].sstate[1]
      return [
        self.fabric[0].sstate[1],
        self.fabric[1].sstate[2],
        self.fabric[7].sstate[2]
      ]



      '''
      return self.fabric[0](
        self.fabric[1](
          -1,
          self.fabric[2](
            -1,
            -1,
            self.fabric[3](
              -1,
              x[0],
              -1
            )[2]
          )[1],
          -1
        )[0],
        -1,
        self.fabric[5](
          -1,
          self.fabric[4](
            -1,
            -1,
            x[1]
          )[1],
          -1
        )[2]
        
      )[1]
      '''
 
# create the model 
model = one_layer_net()
 
def binary_cmos_clipper () :
 with torch.no_grad():
  for pp in model.parameters() :
    if pp.requires_grad :
      if len(pp) > 1 :
        for p in pp :
          if p <= 0.0 :
            torch.nn.init.normal_(p, mean=-1.0, std=0.00001)
          if p > 0.0 :
            torch.nn.init.normal_(p, mean=1.0, std=0.00001)
      else :
        p = pp
        if p <= 0.0 :
          torch.nn.init.normal_(p, mean=-1.0, std=0.00001)
        if p > 0.0 :
          torch.nn.init.normal_(p, mean=1.0, std=0.00001)


#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.002)

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.3)

# This thing trains the 9 weights in 3 controllers of one MUX that select between 0, 1, A, B, ~A, ~B.
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

# To implement custom backprop flow
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, weight_decay=0.01, momentum=0.9)

# HW-optimized
#optimizer = torch.optim.SGD(model.parameters(), lr=0.004, weight_decay=0.01, momentum=0.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.01, momentum=0.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=0.01, momentum=0.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01, momentum=0.9)

# Maintains weights
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
#scheduler = MultiStepLR(optimizer, milestones=[50,100,150,175], gamma=0.25)

# Define the training loop
# Converges on AdamW 0.1

#epochs = 400
epochs = 400 #122
cost = []


for epoch in range(epochs):
    epoch = epoch + 1

    network = 0.0

    #i = random.randrange(len(X))
    for i in range(len(X)) :
      x = X[i]
      y = Y[i]

      for j in range(2) : # average noise over 5 iterations
      #for i in np.random.permutation(range(len(X))) :

        noise = 1.0 # epoch * (1.0 / epoch) # 1.0
        yhat = model(x, 0, noise)

        # Least squares loss
        network = network + torch.square(torch.sub(y[0], yhat[0]))
        network = network + torch.square(torch.sub(y[1], yhat[1]))
        network = network + torch.square(torch.sub(bit_not(x[1]), yhat[2]))
        # Squared hinge loss
        #network = network + torch.square(torch.max(torch.tensor(0.0), 1.0 - 0.125 * y * yhat))

    network.backward()

    lr1 = optimizer.param_groups[0]["lr"]
    print('Epoch:', epoch, 'Loss:', network.item(), 'Learning Rate:', lr1)

    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    #scheduler.step()


    optimizer.zero_grad()

    #if network.item() < 0.01 :
    #  binary_cmos_clipper() 

    cost.append(network.item())

for pp in model.parameters() :
  print(pp) # model.luts[0].weights)


# Final clipper to -1 or +1 only
binary_cmos_clipper() 

print('After binarization:')
for pp in model.parameters() :
  print(pp) # model.luts[0].weights)

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

for i in range(4) :
  x = X[i]
  y_hat = model(X[i], 0, 0.0)
  print(f'A x0: { nn_to_py(x[0].item()) },\tx1: { nn_to_py(x[1].item()) },\ty_hat: { y_hat[0].item() }\ty_should: { Y[i][0].item() }')
  print(f'B x0: { nn_to_py(x[0].item()) },\tx1: { nn_to_py(x[1].item()) },\ty_hat: { y_hat[1].item() }\ty_should: { Y[i][1].item() }')

