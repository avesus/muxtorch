import torch
import torch.nn as nn

# To load the plain serialized model
from collections import OrderedDict

from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import math
import random

import matplotlib.pyplot as plt
import warnings


tensor = torch.tensor

'''
def square (a) :
  return 2 ** (2.0 * math.log2(a))

def mul (a, b) :
  return -0.5 * ( (a - b) * (a - b) - a * a - b * b)
  #return -0.5 * ( square(abs(a - b)) - square(a) - square(b))
'''

def relu_tanh(x, a) :

  try:
    with warnings.catch_warnings():
        warnings.filterwarnings("error")  # Convert warnings to errors
        exp_value = np.exp(-a * x)
  except RuntimeWarning as e:
    exp_value = np.exp(-a * x)
    if math.isinf(exp_value) :
      return 0.0
    print(e, exp_value, x * a)
    
  result = 1 / (1 + exp_value)

  return result

def tanh(x):
    return np.tanh(x)


def sigmoid (x) :
  # return relu_tanh(x, 10000)

  return 1 / (1 + np.exp(-10000 * x))

  #return 1 /(1 + 1 / np.exp(10000.0 * x))
  # return 1 /(1 + 1 / np.exp(10000.0 * x))
  # return 1 /(1 + 1 / np.exp(1000.0 * x * x))

'''
# Generate x values
x = np.linspace(-10, 10, 1000)

# Calculate y values using tanh function
y = relu_tanh(x, 500)
# y = sigmoid(x)

# Plot the tanh function
plt.plot(x, y)
plt.title('Hyperbolic Tangent (tanh) Function')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True)
plt.show()
'''

def dac (binary_value) :
  factor = 1.0
  # factor = len(binary_value)
  total = 0.0
  for bit in binary_value :
    bit = (bit + 1.0) * 0.5
    total = total + bit * factor
    factor = factor * 2.0

  return total

def adc (real_value, nps=False, range_bits=2) :
  n = 2 ** range_bits
  x = real_value + 0.5
  # x = torch.add(real_value, 0.5)
  bits = []
  for i in range(range_bits) :
    n = 0.5 * n
    diff = x - n

    # outside of training:
    if nps :
      bit = sigmoid(diff)

    # inside of training:
    else :
      bit = torch.nn.functional.sigmoid(50.0 * diff)

    one_bit_dac = bit * n
    x = x - one_bit_dac
    bits.append(bit * 2.0 - 1.0)

  bits.reverse()
  return bits

for i in range(4) :
  print(i, adc(i, nps=True), dac(adc(i, nps=True)))

def observation (control) :
  '''
  Sample from the real-world process
  Training data sample from the unknown world distribution.
  '''

  # control is the vector of all variables we kept fixed while
  # performing the real world observation
  X = control
  # Real world model is an external natural data source
  # When we perform measurement on the real distribution, we can only sample
  # some random values from it
  Y = real_world_process(X)
  return Y

# Permutation of A, B, and CARRY_IN
'''
X = torch.tensor([[-1,-1,-1], [-1,1,-1], [1,-1,-1], [1,1,-1],
                  [-1,-1,1], [-1,1,1], [1,-1,1], [1,1,1]])
'''

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

  return ret_val

# A + B + CARRY_IN
# Y = torch.tensor([dataset(x) for x in X])

ADDER_BITS = 2

X = torch.tensor([range(2 * ADDER_BITS) for r in range((2 ** ADDER_BITS) ** 2)])
i = 0
for a in range(2 ** ADDER_BITS) :
  for b in range(2 ** ADDER_BITS) :
    # print(a, b, dac(adc(a)), dac(adc(b)))
    X[i] = torch.concat((torch.tensor(adc(a, nps=True)), torch.tensor(adc(b, nps=True))))
    x = X[i]
    #print(x[0:2], 'should be', adc(a, nps=True), 'and', x[2:4], 'should be', adc(b, nps=True), 'at', a, b)
    # , dac(x[0:3]), dac(x[4:7]))
    i = i + 1

def adder_dataset (x) :
  a = dac(x[0:2])
  b = dac(x[2:4])
  y = a + b
  return y
  # return adc(y)

Y = torch.tensor([adder_dataset(x) for x in X])


for i in range(len(X)) :
  x = X[i]
  print(dac(x[0:2]), dac(x[2:4]), 'y_should', Y[i], dac(x[0:2]) + dac(x[2:4]))


#sdferv


def Mux (s, a, b) :
  return 0.5 * (s * a - s * b + a + b)

'''
Switchbox layer is made of M * 3 + N switches, each of which
is a simple cascade of multiplexors similar to FPGA LUTn,
but instead of n LUT logic inputs they have n weights,
and instead of 2^n LUT entries they have (2^n - 1) inputs from the previous layer,
and one additional weight that is fed instead of the 1 missing input.
Therefore, the input dimension is defined by the number of weights that control
the switchbox: (w - 1) weights control the switchbox and 1 weight is fed as a trainable constant,
when the (w - 1) weights are trained to be all logic "0".
Possible MUX inputs: 2^(w - 1), including 1 reserved constant input.
Input dimension = 2^(w - 1) - 1.
Compute layer is made of M multiplexers without trainable weights
'''

def Cmos (x) :
  #return x
  
  #return 4.0 * torch.nn.functional.sigmoid(50.0 * x) - 2.0
  
  return 2.0 * torch.nn.functional.sigmoid(4.0 * x) - 1.0

class SwitchboxElement(nn.Module) :
  def __init__ (self, num_of_weights, layer_id) :
    super().__init__()

    self.layer_id = layer_id

    self.num_selector_bits = num_of_weights - 1

    self.trained_const_weight = nn.Parameter(torch.zeros(1))
    self.mux_selector_weights = nn.Parameter(torch.zeros(self.num_selector_bits))
    # TODO: divide by sqrt(2 * num_layers)
    #torch.nn.init.normal_(self.trained_const_weight, mean=0, std=0.02)
    #torch.nn.init.normal_(self.mux_selector_weights, mean=0, std=0.02)
    torch.nn.init.normal_(self.trained_const_weight, mean=0, std=0.005)
    torch.nn.init.normal_(self.mux_selector_weights, mean=0, std=0.005)

  def _sel (self, sel) :
    p = self.mux_selector_weights[sel]
    noise = torch.square(1.0 - torch.abs(p)) * np.random.normal() * 0.125
    return p + noise

  def _const (self) :
    p = self.trained_const_weight
    noise = torch.square(1.0 - torch.abs(p)) * np.random.normal() * 0.125
    return p + noise


  # MUX cascade tree
  def forward (self, x) :
    prev = x
    # extend input with logic zeroes if len(x) + 1 < 2 ** num_selector_bits 
    if len(x) + 1 < 2 ** self.num_selector_bits :
      prev = torch.concat((x, torch.tensor([-1.0 for i in range(2 ** self.num_selector_bits - len(x) - 1)])))

    for sel in range(self.num_selector_bits) :
      mux_layer = self.num_selector_bits - sel - 1
      num_muxes = 2 ** mux_layer
      next = torch.zeros(num_muxes)
      for mux_i in range(num_muxes) :
        # last layer has just 1 MUX

        # the last MUX of the first layer gets input from constant weight to support
        # trainable constant weight (simply logic "0" or "1")
        #try :
        if sel == 0 and mux_i == num_muxes - 1:
          next[mux_i] = Mux(self._sel(sel), prev[mux_i * 2], self._const())
          #next[mux_i] = Mux(Cmos(self.mux_selector_weights[sel]), prev[mux_i * 2], Cmos(self.trained_const_weight))
        else :
          next[mux_i] = Mux(self._sel(sel), prev[mux_i * 2], prev[mux_i * 2 + 1])
          #next[mux_i] = Mux(Cmos(self.mux_selector_weights[sel]), prev[mux_i * 2], prev[mux_i * 2 + 1])
        #except Exception as e :
          # index 3 is out of bounds for dimension 0 with size 3 
          #                                   4             3         0     1       8
          # index 8 is out of bounds for dimension 0 with size 8
          # tensor([-1.0067, -1.0067, -1.0067, -1.0067, -1.0067, -1.0067, -1.0067, -1.0067])
          #                         0              4                3 0 4 8
        #  print(len(prev))
        #  print(e, prev, self.layer_id, self.num_selector_bits, mux_layer, sel, mux_i, num_muxes)
      prev = next

    return prev[0]

class RouterBlock(nn.Module):
  def __init__(self, layer_id: int, N, M, input_dimension):
    super().__init__()
    print('RouterBlock#', layer_id, 'Deep Fanouts:', N, 'Muxes:', M, 'MUX Sources:', M * 3, 'inputs:', input_dimension, 'outputs:', N + M) 

    self.skip_switchboxes = torch.nn.ModuleList()
    self.mux_input_switchboxes = torch.nn.ModuleList()

    # input_dimension = 2^(w - 1) - 1
    # input_dimension + 1 = 2^(w - 1)
    # log2(input_dimension + 1) = w - 1
    # w = 1 + log2(input_dimension + 1)

    switch_tree_weights = int(math.ceil(1 + math.log2(input_dimension + 1)))

    for switchbox in range(N):
      self.skip_switchboxes.append(SwitchboxElement(switch_tree_weights, layer_id))

    for switchbox in range(M * 3):
      self.mux_input_switchboxes.append(SwitchboxElement(switch_tree_weights, layer_id))

    self.N = N
    self.M = M

    # Compute has N passthrough wires that do nothing and M multiplexers.
    # n_in   ---> n_out
    # m_0_in --v
    # m_1_in ---> m_out
    # m_2_in --^
    # M multiplexers reduce the input dimension by 3M
    # N passthrough maintain the number
    # K: number of input dimensions = N + 3M
    # Q: number of output dimensions = N + M

    self.layer_id = layer_id

  def forward(self, x):
    skip = torch.zeros(self.N)
    for skip_i in range(self.N) :
      skip[skip_i] = self.skip_switchboxes[skip_i].forward(x)

    process = torch.zeros(self.M * 3)
    for mux_i in range(self.M * 3) :
      process[mux_i] = self.mux_input_switchboxes[mux_i].forward(x)

    computed = torch.zeros(self.M)
    for m in range(self.M) :
      computed[m] = Mux(process[m * 3], process[m * 3 + 1], process[m * 3 + 2])
      #computed[m] = Cmos(Mux(process[m * 3], process[m * 3 + 1], process[m * 3 + 2]))

    return torch.concat((skip, computed))

class one_layer_net (torch.nn.Module):    

    def __init__ (self):
      super(one_layer_net, self).__init__()

      # N = 0.02

      n_layers = 7

      self.layers = torch.nn.ModuleList()
      for layer_id in range(n_layers):
        
        if layer_id == 0:
          # input:
          self.layers.append(RouterBlock(layer_id=layer_id, N=4, M=3, input_dimension = 2 + 2))
        elif layer_id == n_layers - 1 :
          # output:
          self.layers.append(RouterBlock(layer_id=layer_id, N=0, M=3, input_dimension = 9))
        else :
          self.layers.append(RouterBlock(layer_id=layer_id, N=6, M=3, input_dimension = 9))

    def forward (self, x) :

      h = x
      for layer in self.layers:
        h = layer(h)
      # return dac(Cmos(h))
      # return dac(h)
      return h

# create the model 
predict = one_layer_net()
# predict.load_state_dict(OrderedDict([('layers.0.skip_switchboxes.0.trained_const_weight', tensor([1.])), ('layers.0.skip_switchboxes.0.mux_selector_weights', tensor([-0.9997, -0.9995,  1.0002])), ('layers.0.skip_switchboxes.1.trained_const_weight', tensor([1.])), ('layers.0.skip_switchboxes.1.mux_selector_weights', tensor([-1.0008,  0.9994,  1.0001])), ('layers.0.skip_switchboxes.2.trained_const_weight', tensor([1.])), ('layers.0.skip_switchboxes.2.mux_selector_weights', tensor([-1.0011,  1.0004,  0.9981])), ('layers.0.skip_switchboxes.3.trained_const_weight', tensor([1.])), ('layers.0.skip_switchboxes.3.mux_selector_weights', tensor([-0.9997,  0.9990,  0.9994])), ('layers.0.mux_input_switchboxes.0.trained_const_weight', tensor([-1.])), ('layers.0.mux_input_switchboxes.0.mux_selector_weights', tensor([ 1.0002, -1.0000,  1.0005])), ('layers.0.mux_input_switchboxes.1.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.1.mux_selector_weights', tensor([-0.9991, -0.9991,  1.0007])), ('layers.0.mux_input_switchboxes.2.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.2.mux_selector_weights', tensor([-1.0013, -0.9986,  1.0023])), ('layers.0.mux_input_switchboxes.3.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.3.mux_selector_weights', tensor([1.0009, 0.9999, 1.0001])), ('layers.0.mux_input_switchboxes.4.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.4.mux_selector_weights', tensor([ 1.0008, -0.9997,  1.0004])), ('layers.0.mux_input_switchboxes.5.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.5.mux_selector_weights', tensor([0.9998, 1.0009, 0.9995])), ('layers.0.mux_input_switchboxes.6.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.6.mux_selector_weights', tensor([-0.9976, -0.9978, -0.9990])), ('layers.0.mux_input_switchboxes.7.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.7.mux_selector_weights', tensor([1.0022, 1.0008, 1.0006])), ('layers.0.mux_input_switchboxes.8.trained_const_weight', tensor([1.])), ('layers.0.mux_input_switchboxes.8.mux_selector_weights', tensor([-0.9993, -0.9986,  0.9997])), ('layers.1.skip_switchboxes.0.trained_const_weight', tensor([1.])), ('layers.1.skip_switchboxes.0.mux_selector_weights', tensor([1.0001, 0.9973, 0.9994, 1.0000])), ('layers.1.skip_switchboxes.1.trained_const_weight', tensor([1.])), ('layers.1.skip_switchboxes.1.mux_selector_weights', tensor([-1.0001,  1.0013,  1.0007,  0.9997])), ('layers.1.skip_switchboxes.2.trained_const_weight', tensor([1.])), ('layers.1.skip_switchboxes.2.mux_selector_weights', tensor([ 0.9988, -0.9999, -0.9999,  0.9999])), ('layers.1.skip_switchboxes.3.trained_const_weight', tensor([1.])), ('layers.1.skip_switchboxes.3.mux_selector_weights', tensor([ 0.9997,  1.0005, -1.0005,  0.9994])), ('layers.1.skip_switchboxes.4.trained_const_weight', tensor([1.])), ('layers.1.skip_switchboxes.4.mux_selector_weights', tensor([ 1.0001,  1.0001, -0.9992,  1.0007])), ('layers.1.skip_switchboxes.5.trained_const_weight', tensor([1.])), ('layers.1.skip_switchboxes.5.mux_selector_weights', tensor([ 1.0012, -0.9990,  1.0014,  0.9989])), ('layers.1.mux_input_switchboxes.0.trained_const_weight', tensor([1.])), ('layers.1.mux_input_switchboxes.0.mux_selector_weights', tensor([ 1.0007, -1.0022, -1.0015,  1.0000])), ('layers.1.mux_input_switchboxes.1.trained_const_weight', tensor([1.])), ('layers.1.mux_input_switchboxes.1.mux_selector_weights', tensor([-1.0005,  1.0002, -0.9990,  1.0003])), ('layers.1.mux_input_switchboxes.2.trained_const_weight', tensor([1.])), ('layers.1.mux_input_switchboxes.2.mux_selector_weights', tensor([-1.0009,  0.9995, -1.0005,  1.0001])), ('layers.1.mux_input_switchboxes.3.trained_const_weight', tensor([-1.])), ('layers.1.mux_input_switchboxes.3.mux_selector_weights', tensor([-0.9981, -1.0010,  0.9971,  1.0008])), ('layers.1.mux_input_switchboxes.4.trained_const_weight', tensor([-1.])), ('layers.1.mux_input_switchboxes.4.mux_selector_weights', tensor([0.9986, 0.9978, 1.0007, 0.9999])), ('layers.1.mux_input_switchboxes.5.trained_const_weight', tensor([-1.])), ('layers.1.mux_input_switchboxes.5.mux_selector_weights', tensor([-1.0002, -0.9999, -1.0000,  0.9991])), ('layers.1.mux_input_switchboxes.6.trained_const_weight', tensor([1.])), ('layers.1.mux_input_switchboxes.6.mux_selector_weights', tensor([ 1.0015, -0.9999,  1.0014, -1.0007])), ('layers.1.mux_input_switchboxes.7.trained_const_weight', tensor([-1.])), ('layers.1.mux_input_switchboxes.7.mux_selector_weights', tensor([ 1.0019,  0.9998, -1.0019,  0.9986])), ('layers.1.mux_input_switchboxes.8.trained_const_weight', tensor([1.])), ('layers.1.mux_input_switchboxes.8.mux_selector_weights', tensor([-0.9982,  1.0012, -1.0005,  0.9979])), ('layers.2.mux_input_switchboxes.0.trained_const_weight', tensor([-1.])), ('layers.2.mux_input_switchboxes.0.mux_selector_weights', tensor([1.0007, 1.0007, 0.9998, 1.0003])), ('layers.2.mux_input_switchboxes.1.trained_const_weight', tensor([-1.])), ('layers.2.mux_input_switchboxes.1.mux_selector_weights', tensor([1.0006, 0.9994, 0.9988, 0.9989])), ('layers.2.mux_input_switchboxes.2.trained_const_weight', tensor([-1.])), ('layers.2.mux_input_switchboxes.2.mux_selector_weights', tensor([ 1.0010, -0.9992,  1.0006,  1.0010])), ('layers.2.mux_input_switchboxes.3.trained_const_weight', tensor([1.])), ('layers.2.mux_input_switchboxes.3.mux_selector_weights', tensor([-1.0020,  1.0011,  0.9993,  0.9997])), ('layers.2.mux_input_switchboxes.4.trained_const_weight', tensor([1.])), ('layers.2.mux_input_switchboxes.4.mux_selector_weights', tensor([-0.9992, -1.0005, -0.9989, -1.0011])), ('layers.2.mux_input_switchboxes.5.trained_const_weight', tensor([1.])), ('layers.2.mux_input_switchboxes.5.mux_selector_weights', tensor([-1.0008, -1.0021, -0.9993,  1.0003])), ('layers.2.mux_input_switchboxes.6.trained_const_weight', tensor([1.])), ('layers.2.mux_input_switchboxes.6.mux_selector_weights', tensor([-0.9984, -0.9986, -0.9988, -1.0005])), ('layers.2.mux_input_switchboxes.7.trained_const_weight', tensor([1.])), ('layers.2.mux_input_switchboxes.7.mux_selector_weights', tensor([ 1.0014, -0.9987, -0.9998,  0.9998])), ('layers.2.mux_input_switchboxes.8.trained_const_weight', tensor([1.])), ('layers.2.mux_input_switchboxes.8.mux_selector_weights', tensor([-0.9986,  1.0006,  1.0013,  1.0004]))]))

def binary_cmos_clipper () :
 with torch.no_grad():
  for pp in predict.parameters() :
    if pp.requires_grad :
      if len(pp) > 1 :
        for p in pp :
          if p <= 0.0 :
            torch.nn.init.normal_(p, mean=-1.0, std=0.001)
          if p > 0.0 :
            torch.nn.init.normal_(p, mean=1.0, std=0.001)
      else :
        p = pp
        if p <= 0.0 :
          torch.nn.init.normal_(p, mean=-1.0, std=0.001)
        if p > 0.0 :
          torch.nn.init.normal_(p, mean=1.0, std=0.001)

def cmos_clipper () :
 with torch.no_grad():
  for pp in predict.parameters() :
    if pp.requires_grad :
      if len(pp) > 1 :
        for p in pp :
          if p < 0.05 and p > -0.05 :
            settled_to = 2.0 * random.randrange(2) - 1
            torch.nn.init.normal_(p, mean=settled_to, std=0.01)
          if p > 1.0 :
            torch.nn.init.normal_(p, mean=1.0, std=0.0001)
          elif p < -1.0 :
            torch.nn.init.normal_(p, mean=-1.0, std=0.0001)
          #elif p >= 0.15 :
          #  torch.nn.init.normal_(p, mean=1.0, std=0.01)
          #elif p <= -0.15 :
          #  torch.nn.init.normal_(p, mean=-1.0, std=0.01)
      else :
        p = pp
        if p < 0.05 and p > -0.05 :
          settled_to = 2.0 * random.randrange(2) - 1
          torch.nn.init.normal_(p, mean=settled_to, std=0.01)
        if p > 1.0 :
          torch.nn.init.normal_(p, mean=1.0, std=0.0001)
        elif p < -1.0 :
          torch.nn.init.normal_(p, mean=-1.0, std=0.0001)
        #elif p >= 0.15 :
        #  torch.nn.init.normal_(p, mean=1.0, std=0.01)
        #elif p <= -0.15 :
        #  torch.nn.init.normal_(p, mean=-1.0, std=0.01)

# binary_cmos_clipper()

print('Total model parameter bits:', sum(p.numel() for p in predict.parameters()))
print('LUT would require bits:', ADDER_BITS * (2 **(2 * ADDER_BITS)))

with torch.no_grad():
  print('Current weights:', torch.concat([0.5 * (p + 1.0) for p in predict.parameters() if p.requires_grad]))

'''
Training 2-bit -> 3-bit adder worked on:
optimizer = torch.optim.AdamW(predict.parameters(), lr=0.1)
scheduler = MultiStepLR(optimizer, milestones=[100,300,500,800], gamma=0.5)
MINIBATCH_SIZE = 20
EPOCHS = 50
x = X[i] + 0.01 * np.random.normal()
No CMOS sigmoid
'''

'''
Non-Stochastic AdamW
optimizer = torch.optim.AdamW(predict.parameters(), lr=0.05)
scheduler = MultiStepLR(optimizer, milestones=[100,300,500,800], gamma=0.5)
x = X[i] # + 0.001 * np.random.normal()
loss 7.66
MINIBATCH_SIZE = 20
EPOCHS = 240
'''


#MINIBATCH_SIZE = 3
MINIBATCH_SIZE = 8
EPOCHS = 500

# From scratch AdamW LR=0.1
optimizer = torch.optim.AdamW(predict.parameters(), lr=0.25)
#optimizer = torch.optim.AdamW(predict.parameters(), lr=0.001)
#optimizer = torch.optim.AdamW(predict.parameters(), lr=0.005)
#optimizer = torch.optim.AdamW(predict.parameters(), lr=0.08)

# Fine-tune AdamW LR=0.001 or less
#optimizer = torch.optim.AdamW(predict.parameters(), lr=0.00001)
# To implement custom backprop flow
#optimizer = torch.optim.SGD(predict.parameters(), lr=0.0001, weight_decay=0.0, momentum=0.0)
#optimizer = torch.optim.SGD(predict.parameters(), lr=0.005, weight_decay=0.0, momentum=0.0)

scheduler = MultiStepLR(optimizer, milestones=[100,300,500,800], gamma=0.5)
#scheduler = MultiStepLR(optimizer, milestones=[40,100,150,200], gamma=0.5)


loss_over_epochs = []

reset = 25

# if reset < 25 :
for epoch in range(EPOCHS):
    epoch = epoch + 1

    network = 0.0

    #for minibatch in range(MINIBATCH_SIZE) :
    for j in range(3) : # Average the Gaussian noise of the weights
      for i in range(len(X)) :
        # for i in np.random.permutation(range(len(training_data))) :
        #i = random.randrange(len(X))
        #x = X[i]
        x = X[i]# + 0.1 * np.random.normal()
        #x = X[i]# + 0.001 * np.random.normal()
        y = Y[i] #  + 0.00001 * np.random.normal()


        # forward
        yhat = predict(x)
        # yhat = dac(predict(x))

        binary_yhat0 = yhat[0]
        binary_y0 = adc(y, nps=False, range_bits=3)[0]
        binary_yhat1 = yhat[1]
        binary_y1 = adc(y, nps=False, range_bits=3)[1]
        
        binary_yhat2 = yhat[2]
        binary_y2 = adc(y, nps=False, range_bits=3)[2]

        '''
        binary_yhat3 = yhat[3]
        binary_y3 = adc(y, nps=False, range_bits=5)[3]

        binary_yhat3 = yhat[4]
        binary_y3 = adc(y, nps=False, range_bits=5)[4]
        '''

        
        network = (network
          + torch.square(1.0 * torch.sub(binary_y0, binary_yhat0))
          + torch.square(2.0 * torch.sub(binary_y1, binary_yhat1))
          + torch.square(4.0 * torch.sub(binary_y2, binary_yhat2)))
          # + torch.square(8.0 * torch.sub(binary_y3, binary_yhat3)))
        
        analog_yhat = dac([yhat[0], yhat[1], yhat[2]]) # , yhat[3]])

        ''' 
        print('Learn input', x, 'into output')
        print([binary_y0.item(), binary_y1.item(), binary_y2.item()], y, 'got')
        print([a.item() for a in yhat], analog_yhat)
        '''
        
        #network = network + torch.square(torch.sub(y, analog_yhat))
        '''
        if abs(y) > 0.00000001 :
          network = network + torch.square(torch.sub(1, 1/y * analog_yhat))
        else :
          network = network + torch.square(analog_yhat)
        '''

    '''
    # Penalize weights too far from -1 and +1 (logic "0" and "1")
    for pp in predict.parameters() :
      if pp.requires_grad :
        if len(pp) > 1 :
          for p in pp :
            network = network + torch.square(0.005 * (torch.abs(p) - 1.0))
        else :
          p = pp
          network = network + torch.square(0.005 * (torch.abs(p) - 1.0))
    '''
    # backward
    network.backward()

    lr1 = optimizer.param_groups[0]["lr"]
    print('Epoch:', epoch, 'Loss:', network.item(), 'Learning Rate:', lr1)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # To debug the optimizer
    loss_over_epochs.append(network.item())

    #if network.item() < 1.0 :
    #  binary_cmos_clipper() 
    ''' 
    reset = reset - 1
    if reset == 0 :
      reset = 25
      binary_cmos_clipper() 
    '''


# Debug the weights before the clipper
print('Learned weights:', predict.state_dict())
with torch.no_grad():
  print('Complete weights list:', [w.item() for w in torch.concat([0.5 * (p + 1.0) for p in predict.parameters() if p.requires_grad])])


# Final clipper to -1 or +1 only
binary_cmos_clipper() 

# plot the loss over epochs
plt.plot(loss_over_epochs)
plt.xlabel('Epochs')
plt.title('Least Squares Loss')
plt.show()

# Validate on the entire range for small bit counts
total_error = 0.0
with torch.no_grad():
  for i in range(len(X)) :
    x = X[i]
    y = Y[i]
    # yhat = adc(predict(x), nps=False)
    yhat = predict(x)
    # print(dac(yhat), dac([yhat[0], yhat[1], yhat[2], yhat[3]]))
    binary_yhat0 = Cmos(yhat[0])
    binary_y0 = adc(y, nps=True, range_bits=3)[0]
    binary_yhat1 = Cmos(yhat[1])
    binary_y1 = adc(y, nps=True, range_bits=3)[1]
    
    binary_yhat2 = Cmos(yhat[2])
    binary_y2 = adc(y, nps=True, range_bits=3)[2]
    '''
    binary_yhat3 = yhat[3]
    binary_y3 = adc(y, nps=True)[3]
    '''
    total_error = (total_error
      + torch.square(torch.sub(binary_y0, binary_yhat0))
      + 2.0 * torch.square(torch.sub(binary_y1, binary_yhat1))
      + 4.0 * torch.square(torch.sub(binary_y2, binary_yhat2)))
      # + torch.square(torch.sub(binary_y3, binary_yhat3)))
    # total_error = total_error + torch.square(torch.sub(torch.tensor(Y[i]), torch.tensor(yhat))).item()
    analog_yhat = dac([yhat[0], yhat[1], yhat[2]]) #, yhat[3]])

    
    print('For inputs', dac(x[0:2]).item(), dac(x[2:4]).item())
    print('Should be:')
    # print([binary_y0.item(), binary_y1.item(), binary_y2.item(), binary_y3.item()], y, 'got:')
    print([binary_y0.item(), binary_y1.item(), binary_y2.item()], y, 'got:')
    print([a.item() for a in yhat], analog_yhat, '\n')
    
    # total_error = total_error + (analog_yhat - y) ** 2

    '''
    print(dac(x[0:4]).item(), dac(x[4:8]).item(), [yhat[0], yhat[1], yhat[2], yhat[3]], 'yhat y_should', dac([yhat[0], yhat[1], yhat[2], yhat[3]]),
      # dac(yhat).item(),
      Y[i].item(), 'error',
      abs(dac([yhat[0], yhat[1], yhat[2], yhat[3]]) - Y[i]).item()
      #abs(dac(yhat) - Y[i]).item()
      )
    '''
print('Total loss', total_error * (MINIBATCH_SIZE / len(X)))


with torch.no_grad():
  all_weights = torch.concat([0.5 * (p + 1.0) for p in predict.parameters() if p.requires_grad])
  total_weight_noise = 0.0
  noisy_weights = []
  for weight in all_weights :
    if weight < 0.5 :
      weight_noise = abs(0.0 - weight)
    else :
      weight_noise = abs(1.0 - weight)
    if weight_noise > 0.45 :
      noisy_weights.append(weight)
    total_weight_noise = total_weight_noise + weight_noise
  print('Total weights noise:', total_weight_noise, [w.item() for w in noisy_weights])



