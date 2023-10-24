import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import math
import random
from scipy.signal import max_len_seq

prng_shift_reg = None
for i in range(30) :
  seq, prng_shift_reg = max_len_seq(4, prng_shift_reg, 1)
  print(seq)


prng_shift_reg = None

def prng() :
  #return np.random.normal()
  global prng_shift_reg
  seq, prng_shift_reg = max_len_seq(4, prng_shift_reg, 1)
  return 2.0 * seq[0] - 1.0

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

mode = None

def Mux (s, a, b) :
  global mode
  if mode :
    return b if s > 0 else a

  val = 0.5 * (s * b - s * a + a + b)
  # print(s, a, b, val)
  return val

mode = True
print('Test MUX')
print(Mux(-1, -1, -1))
print(Mux(-1, 1, -1))
print(Mux(-1, -1, 1))
print(Mux(-1, 1, 1))
print(Mux(1, -1, -1))
print(Mux(1, 1, -1))
print(Mux(1, -1, 1))
print(Mux(1, 1, 1))

mode = False

class LUT2 (torch.nn.Module) :
  def __init__ (self) :
    super().__init__()

    self.weights = torch.nn.Parameter(torch.zeros(4))
    #torch.nn.init.constant_(self.weights, -1.0)

  def forward (self, a, b, noise) :

    return Mux(b,
      Mux(a,
        self.weights[0] + torch.abs(torch.sub(1.0, torch.abs(self.weights[0]))) * prng() * noise,
        self.weights[1] + torch.abs(torch.sub(1.0, torch.abs(self.weights[1]))) * prng() * noise
      ),
      Mux(a,
        self.weights[2] + torch.abs(torch.sub(1.0, torch.abs(self.weights[2]))) * prng() * noise,
        self.weights[3] + torch.abs(torch.sub(1.0, torch.abs(self.weights[3]))) * prng() * noise
      ))

# 3 x LUT-2
class Triangle (torch.nn.Module):    

  def __init__ (self):
    super(Triangle, self).__init__()
    self.luts = torch.nn.ModuleList()
    for lut in range(3):
      self.luts.append(LUT2())

  def forward (self, x0, x1, x2, noise) :
    return [
      self.luts[0].forward(x1, x2, noise),
      self.luts[1].forward(x0, x2, noise),
      self.luts[2].forward(x0, x1, noise) ]


# 1 x LUT-2, rotation, full fan-out, intersections if rot = 00
class Triangle2 (torch.nn.Module):    

  def __init__ (self):
    super(Triangle2, self).__init__()
    self.rot = torch.nn.Parameter(torch.zeros(2))
    self.lut = torch.nn.Parameter(torch.zeros(4))
    #torch.nn.init.constant_(self.rot, -1.0)
    #torch.nn.init.constant_(self.lut, -1.0)

  def forward (self, x0, x1, x2, noise=0.0) :

    lut0 = self.lut[0] + torch.abs(torch.sub(1.0, torch.abs(self.lut[0]))) * prng() * noise
    lut1 = self.lut[1] + torch.abs(torch.sub(1.0, torch.abs(self.lut[1]))) * prng() * noise
    lut2 = self.lut[2] + torch.abs(torch.sub(1.0, torch.abs(self.lut[2]))) * prng() * noise
    lut3 = self.lut[3] + torch.abs(torch.sub(1.0, torch.abs(self.lut[3]))) * prng() * noise

    rot0 = self.rot[0] + torch.abs(torch.sub(1.0, torch.abs(self.rot[0]))) * prng() * noise
    rot1 = self.rot[1] + torch.abs(torch.sub(1.0, torch.abs(self.rot[1]))) * prng() * noise

    # inputs into the LUT2
    a = self.a = Mux(rot0, x1, x0)
    b = self.b = Mux(rot0,
      x2,
      Mux(rot1, x2, x1))

    '''
    rot
    10=02
    01=12
    11=01
        rot0 rot1
    #a = 0    0    x1
    #b = 0    0    x2

    a = 1    0    x0
    b = 1    0    x2

    a = 0    1    x1
    b = 0    1    x2

    a = 1    1    x0
    b = 1    1    x1

    '''

    # Output of the LUT2 is fan out if rot0 and rot1 are not logic "0"
    lut2_out = self.lut2_out = Mux(b,
      Mux(a,
        lut0,
        lut1
      ),
      Mux(a,
        lut2,
        lut3
      ))

    # outputs are routed as an intersection if rot0 and rot1 are logic "0",
    # and lut0 is 0
    out0 = Mux(rot0,
      Mux(rot1,
        Mux(lut0, x1, x2),
        lut2_out
      ),
      lut2_out)

    out1 = Mux(rot0,
      Mux(rot1,
        Mux(lut0, x2, x0),
        lut2_out
      ),
      lut2_out)

    out2 = Mux(rot0,
      Mux(rot1,
        Mux(lut0, x0, x1),
        lut2_out
      ),
      lut2_out)
    
    return [
      out0,
      out1,
      out2 ]

# Test
with torch.no_grad():
  t1 = Triangle2()
  t1.lut[0] = -1
  t1.lut[1] = -1
  t1.lut[2] = -1
  t1.lut[3] = -1

  t1.rot[0] = -1
  t1.rot[1] = -1

  for x0 in range(-1, 3, 2) :
    for x1 in range(-1, 3, 2) :
      for x2 in range(-1, 3, 2) :
        n = t1(x0, x1, x2)
        print(x0, x1, x2, 'outs', n)
        assert(abs(n[0] - x1) < 0.125)
        assert(abs(n[1] - x2) < 0.125)
        assert(abs(n[2] - x0) < 0.125)

  t1.lut[0] = 1
  for x0 in range(-1, 3, 2) :
    for x1 in range(-1, 3, 2) :
      for x2 in range(-1, 3, 2) :
        n = t1(x0, x1, x2)
        print(x0, x1, x2, 'outs', n)
        assert(abs(n[0] - x2) < 0.125)
        assert(abs(n[1] - x0) < 0.125)
        assert(abs(n[2] - x1) < 0.125)

  # rot0 rot1 = a b
  #    10=02
  #    01=12
  #    11=01
  t1.rot[0] = 1
  t1.rot[1] = -1

  for x0 in range(-1, 3, 2) :
    for x1 in range(-1, 3, 2) :
      for x2 in range(-1, 3, 2) :
        n = t1(x0, x1, x2)

        print(x0, x1, x2, 'outs', n, 'lut2_out', t1.lut2_out)
        assert(abs(n[0] - t1.lut2_out) < 0.125)
        assert(abs(n[1] - t1.lut2_out) < 0.125)
        assert(abs(n[2] - t1.lut2_out) < 0.125)
        assert(abs(n[2] - t1.lut2_out) < 0.125)
        assert(abs(t1.a - x0) < 0.125)
        assert(abs(t1.b - x2) < 0.125)

  # rot0 rot1 = a b
  #    10=02
  #    01=12
  #    11=01
  t1.rot[0] = -1
  t1.rot[1] = 1

  for x0 in range(-1, 3, 2) :
    for x1 in range(-1, 3, 2) :
      for x2 in range(-1, 3, 2) :
        n = t1(x0, x1, x2)

        print(x0, x1, x2, 'outs', n, 'lut2_out', t1.lut2_out)
        assert(abs(n[0] - t1.lut2_out) < 0.125)
        assert(abs(n[1] - t1.lut2_out) < 0.125)
        assert(abs(n[2] - t1.lut2_out) < 0.125)
        assert(abs(n[2] - t1.lut2_out) < 0.125)
        assert(abs(t1.a - x1) < 0.125)
        assert(abs(t1.b - x2) < 0.125)

  # rot0 rot1 = a b
  #    10=02
  #    01=12
  #    11=01
  t1.rot[0] = 1
  t1.rot[1] = 1

  for x0 in range(-1, 3, 2) :
    for x1 in range(-1, 3, 2) :
      for x2 in range(-1, 3, 2) :
        n = t1(x0, x1, x2)

        print(x0, x1, x2, 'outs', n, 'lut2_out', t1.lut2_out)
        assert(abs(n[0] - t1.lut2_out) < 0.125)
        assert(abs(n[1] - t1.lut2_out) < 0.125)
        assert(abs(n[2] - t1.lut2_out) < 0.125)
        assert(abs(n[2] - t1.lut2_out) < 0.125)
        assert(abs(t1.a - x0) < 0.125)
        assert(abs(t1.b - x1) < 0.125)

class LUT3 (torch.nn.Module) :
  def __init__ (self) :
    super().__init__()

    self.weights = torch.nn.Parameter(torch.zeros(8))
    #torch.nn.init.constant_(self.weights, -1.0)

  def forward (self, a, b, c, noise) :

    return Mux(c,
      Mux(b,
        Mux(a,
          self.weights[0] + torch.abs(torch.sub(1.0, torch.abs(self.weights[0]))) * prng() * noise,
          self.weights[1] + torch.abs(torch.sub(1.0, torch.abs(self.weights[1]))) * prng() * noise
        ),
        Mux(a,
          self.weights[2] + torch.abs(torch.sub(1.0, torch.abs(self.weights[2]))) * prng() * noise,
          self.weights[3] + torch.abs(torch.sub(1.0, torch.abs(self.weights[3]))) * prng() * noise
        )),
      Mux(b,
        Mux(a,
          self.weights[4] + torch.abs(torch.sub(1.0, torch.abs(self.weights[4]))) * prng() * noise,
          self.weights[5] + torch.abs(torch.sub(1.0, torch.abs(self.weights[5]))) * prng() * noise
        ),
        Mux(a,
          self.weights[6] + torch.abs(torch.sub(1.0, torch.abs(self.weights[6]))) * prng() * noise,
          self.weights[7] + torch.abs(torch.sub(1.0, torch.abs(self.weights[7]))) * prng() * noise
        )))


# 3 x LUT-3
class Triangle3 (torch.nn.Module):

  def __init__ (self):
    super(Triangle3, self).__init__()
    self.luts = torch.nn.ModuleList()
    for lut in range(3):
      self.luts.append(LUT3())

  def forward (self, x0, x1, x2, noise) :
    return [
      self.luts[0].forward(x1, x2, x0, noise),
      self.luts[1].forward(x0, x2, x1, noise),
      self.luts[2].forward(x0, x1, x2, noise) ]


# Fabric
class one_layer_net (torch.nn.Module):    

    def __init__ (self):
        super(one_layer_net, self).__init__()
        self.fabric = torch.nn.ModuleList()
        for i in range(15) :
          self.fabric.append(Triangle())

    def forward (self, x, dummy1, noise=0.0) :


      #RNN_STEPS = 8 # learns with 0.125 noise
      RNN_STEPS = 16 # learns with 0.125 noise
      #RNN_STEPS = 20 # learns with 0.125 noise
      #RNN_STEPS = 130

      prev = []
      next = []
      for i in range(len(self.fabric)) :
        prev.append([ torch.tensor(-1), torch.tensor(-1), torch.tensor(-1) ])
        next.append([ torch.tensor(-1), torch.tensor(-1), torch.tensor(-1) ])

      a = x[0]
      b = x[1]

      for i in range(RNN_STEPS) :

        next[0] = self.fabric[0](prev[1][0], prev[12][1], prev[5][2], noise)
        next[1] = self.fabric[1](prev[0][0],  prev[2][1], prev[9][2], noise)
        next[2] = self.fabric[2](prev[7][0],  prev[1][1], prev[3][2], noise)
        next[3] = self.fabric[3](prev[4][0],  prev[6][1], prev[2][2], noise)
        next[4] = self.fabric[4](prev[3][0],  prev[5][1], b, noise)
        next[5] = self.fabric[5](prev[8][0],  prev[4][1], prev[0][2], noise)
        next[6] = self.fabric[6](-1,          prev[3][1], a, noise)
        next[7] = self.fabric[7](prev[2][0], -1,         -1, noise)
        next[8] = self.fabric[8](prev[5][0], prev[14][1], -1, noise)

        next[9] = self.fabric[9](-1,            prev[10][1], prev[1][2], noise)
        next[10] = self.fabric[10](prev[11][0],  prev[9][1], -1, noise)
        next[11] = self.fabric[11](prev[10][0], -1,          prev[12][2], noise)
        next[12] = self.fabric[12](prev[13][0],  prev[0][1], prev[11][2], noise)

        next[13] = self.fabric[13](prev[12][0], -1, prev[14][2], noise)
        next[14] = self.fabric[14](-1, prev[8][1], prev[13][2], noise)

        #if i > 2 :
        #  a = torch.tensor(-1)
        #  b = torch.tensor(-1)

        for j in range(len(self.fabric)) :
          prev[j][0] = next[j][0]
          prev[j][1] = next[j][1]
          prev[j][2] = next[j][2]

      return [
        prev[11][1],
        prev[10][2],
        #prev[1][1],
        #prev[0][2],
        prev[7][2]
      ]

 
# create the model 
model = one_layer_net()
 
def binary_cmos_clipper () :
 with torch.no_grad():
  for pp in model.parameters() :
    if pp.requires_grad :
      if len(pp) > 1 :
        for p in pp :
          if p <= 0.0 :
            torch.nn.init.constant_(p, -1.0)
          if p > 0.0 :
            torch.nn.init.constant_(p, 1.0)
      else :
        p = pp
        if p <= 0.0 :
          torch.nn.init.constant_(p, -1.0)
        if p > 0.0 :
          torch.nn.init.constant_(p, 1.0)


#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.002)

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.3)

# This thing trains the 9 weights in 3 controllers of one MUX that select between 0, 1, A, B, ~A, ~B.
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.025)

# To implement custom backprop flow
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, weight_decay=0.01, momentum=0.9)

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
#epochs = 220 #122
epochs = 130 #122
cost = []

reset = False
clip = 0
#noise = 0.001
#noise = 0.015
#noise = 0.03
#noise = 0.09
#noise = 0.0625
noise = 0.125
#noise = 0.25
#noise = 0.3333
#noise = 0.5
#noise = 2.0
#noise = 4.0
#noise = 0.0

for epoch in range(epochs):
    epoch = epoch + 1

    network = 0.0

    #i = random.randrange(len(X))
    for i in range(len(X)) :
      x = X[i]
      y = Y[i]

      #for j in range(2) : # average noise over 5 iterations
      for j in range(1) : # average noise over 5 iterations
      #for i in np.random.permutation(range(len(X))) :

        #              noise = epoch * (1.0 / epoch) # 1.0
        yhat = model(x, 0, noise)

        # Least squares loss
        #print(y[0], yhat[0])
        network = network + torch.square(torch.sub(y[0], yhat[0]))
        network = network + torch.square(torch.sub(y[1], yhat[1]))
        #network = network + torch.square(torch.sub(py_to_nn(bit_not(nn_to_py(x[1]))), yhat[2]))
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

    '''
    if clip == 15 :
      clip = 0
      noise *= 1.05
      #binary_cmos_clipper()

    clip += 1

    
    if network.item() < 2.00 :
      if not reset :
        reset = True
        binary_cmos_clipper()
    '''
    

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

mode = True

for i in range(4) :
  x = X[i]
  y_hat = model(X[i], 0)
  print(f'A x0: { nn_to_py(x[0].item()) },\tx1: { nn_to_py(x[1].item()) },\ty_hat: { y_hat[0].item() }\ty_should: { Y[i][0].item() }')
  print(f'B x0: { nn_to_py(x[0].item()) },\tx1: { nn_to_py(x[1].item()) },\ty_hat: { y_hat[1].item() }\ty_should: { Y[i][1].item() }')
  print(f'Inv x1: { nn_to_py(x[1].item()) },\ty_hat: { y_hat[2].item() }')

