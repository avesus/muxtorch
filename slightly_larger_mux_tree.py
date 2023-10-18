import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import math
import random

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
  return py_to_nn(bit_not(a ^ b))
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

class one_layer_net (torch.nn.Module):    

    def __init__ (self):
        super(one_layer_net, self).__init__()

        L = 9
        N = 0.02

        self.l1 = torch.nn.Linear(1, L, bias=False)

        ###
        ### Analog of torch.nn.init.xavier_uniform_(self.l1.weight)
        ###
        torch.nn.init.normal_(self.l1.weight, mean=0, std=0.02)
        #self.l1.weight.data.fill_(0)
        #torch.nn.init.normal_(self.l1.weight, mean=0, std=N/math.sqrt(L))

        '''
        # Routable MUX:
        selC = Mux(w0,
          Mux(w1, x[0], x[1]),
          Mux(w1, -1.0, 1.0))

        selA = Mux(w2,
          Mux(w3, x[0], x[1]),
          Mux(w3, -1.0, 1.0))

        selB = Mux(w4,
          Mux(w5, x[0], x[1]),
          Mux(w5, -1.0, 1.0))
        '''
        '''
        # A AND B:
        torch.nn.init.normal_(self.l1.weight[0][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[1][0], mean=-1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[2][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[3][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[4][0], mean=-1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[5][0], mean=1, std=0.0001)
        
        # A OR B:
        torch.nn.init.normal_(self.l1.weight[0][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[1][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[2][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[3][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[4][0], mean=1, std=0.0001)
        torch.nn.init.normal_(self.l1.weight[5][0], mean=-1, std=0.0001)
        '''
        print(self.l1.state_dict())

        # self.l1.weight[0][0].retain_grad()
        self.l1.weight.retain_grad()

    def forward (self, x) :

      # Noise bases if weights are too far from abs(1)
      # k = 1.0 #  + 0.05 * np.random.normal()

      noise = torch.zeros(self.l1.weight.shape[0])

      j = 0
      
      for pp in self.parameters() :
        if pp.requires_grad :
          if len(pp) > 1 :
            for p in pp :
              #noise[j] = torch.square(1.0 - torch.abs(p))
              # Much much closer to 1
              noise[j] = abs(1.0 - abs(p))
              j = j + 1
          else :
            p = pp
            #noise[j] = torch.square(1.0 - torch.abs(p))
            noise[j] = abs(1.0 - abs(p))
            j = j + 1

      noise_scaler = 2.0

      w0 = self.l1.weight[0][0] + noise[0] * np.random.normal() * 0.125 * noise_scaler
      # w0.register_hook(on_w0_grad)
      w1 = self.l1.weight[1][0] + noise[1] * np.random.normal() * 0.125 * noise_scaler
      w2 = self.l1.weight[2][0] + noise[2] * np.random.normal() * 0.125 * noise_scaler
      w3 = self.l1.weight[3][0] + noise[3] * np.random.normal() * 0.125 * noise_scaler
      w4 = self.l1.weight[4][0] + noise[4] * np.random.normal() * 0.125 * noise_scaler
      w5 = self.l1.weight[5][0] + noise[5] * np.random.normal() * 0.125 * noise_scaler
      w6 = self.l1.weight[6][0] + noise[6] * np.random.normal() * 0.125 * noise_scaler
      w7 = self.l1.weight[7][0] + noise[7] * np.random.normal() * 0.125 * noise_scaler
      w8 = self.l1.weight[8][0] + noise[8] * np.random.normal() * 0.125 * noise_scaler

      # self.w0 = w0

      a = x[0]
      b = x[1]
      inv_a = Mux(a, 1, -1)
      inv_b = Mux(b, 1, -1)

      # Routable MUX:
      w0_mux_a = Mux(w1,
          Mux(w2, a, inv_a),
          Mux(w2, b, inv_b))

      w0_mux_b = Mux(w1, 1.0, -1.0) 

      selC = Mux(w0,
        w0_mux_a,
        w0_mux_b)

      selA = Mux(w3,
        Mux(w4,
          Mux(w5, a, inv_a),
          Mux(w5, b, inv_b)),
        Mux(w4, 1.0, -1.0))

      selB = Mux(w6,
        Mux(w7,
          Mux(w8, a, inv_a),
          Mux(w8, b, inv_b)),
        Mux(w7, 1.0, -1.0))

      # Gradient to w0:
      self.w0_grad = 0.5 * (w0_mux_b.item() - w0_mux_a.item()) * (0.5 * (selB.item() - selA.item()))

      return Mux(selC, selA, selB)
      #return torch.tensor(Mux(selC, selA, selB), requires_grad=True)
      
      # LUT2:
      return Mux(x[1],
        Mux(x[0], self.l1.weight[0][0], self.l1.weight[1][0]),
        Mux(x[0], self.l1.weight[2][0], self.l1.weight[3][0]))

      # Adder:
      m00 = Mux(x[0], self.l1.weight[0][0], self.l1.weight[1][0])
      m01 = Mux(x[0], self.l1.weight[2][0], self.l1.weight[3][0])
      m10 = Mux(x[1], m00, m01)
      return m10

 
# create the model 
model = one_layer_net()
 

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.002)

# optimizer = torch.optim.AdamW(model.parameters(), lr=0.3)

# This thing trains the 9 weights in 3 controllers of one MUX that select between 0, 1, A, B, ~A, ~B.
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

# To implement custom backprop flow
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, weight_decay=0.01, momentum=0.9)

optimizer = torch.optim.SGD(model.parameters(), lr=0.004, weight_decay=0.01, momentum=0.9)

# Maintains weights
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
#scheduler = MultiStepLR(optimizer, milestones=[200,250,300,350], gamma=0.25)

# Define the training loop
# Converges on AdamW 0.1

epochs = 170
cost = []

sgd = False
if sgd :
  epochs *= 100

epochs *= 3

w0_momentum = 0.0
b = 0.0

next_w = model.l1.weight[0][0].item()
obj_w = model.l1.weight[0][0].item()

for epoch in range(epochs):
    epoch = epoch + 1

    network = 0.0

    # yhat_00 = None
    yhat_00_grad = 0.0
    w0_grad = 0.0
    w00_grad = 0.0

    torch_w0_grad = 0.0

    if sgd :
        i = random.randrange(len(X))
        x = X[i]
        y = Y[i]
        yhat = model(x)
        network = network + torch.square(torch.sub(y, yhat))

    else :
      #i = random.randrange(len(X))
      for i in range(len(X)) :
        x = X[i]
        y = Y[i]

        for j in range(5) : # average noise over 5 iterations
        #for i in np.random.permutation(range(len(X))) :


          yhat = model(x)

          #if i == 0 and j == 0 :
          # Manual gradient:

          yhat_00_grad = 2.0 * (yhat.item() - y.item()) 
          w00_grad += model.w0_grad * yhat_00_grad

          # yhat_00 = yhat.grad

          # Least squares loss
          network = network + torch.square(torch.sub(y, yhat))
          # Squared hinge loss
          # network = network + torch.square(torch.max(torch.tensor(0.0), 1.0 - 0.125 * y * yhat))

    network.backward()
    print('Total gradient of w0', model.l1.weight.grad[0][0].item() - w00_grad) # , torch_w0_grad)

    # w00_grad = model.l1.weight.grad[0][0].item()

    b =       0.9 * b                    + w00_grad + 0.01 * obj_w
    next_w = obj_w - 0.004 * b

    #b =       0.9 * b           +  0.004 * w00_grad + 0.00004 * obj_w
    #next_w = 0.99 * obj_w - b

    #b =       0.9 * b                    + w00_grad + 0.01 * model.l1.weight[0][0].item()
    #next_w = model.l1.weight[0][0].item() - 0.004 * b

    #next_w = 0.99 * model.l1.weight[0][0].item() - b
    #b =       0.9 * b           +  0.004 * w00_grad + 0.00004 * model.l1.weight[0][0].item()

    #next_m = 0.9 * w0_momentum + 0.004 * (w0_grad + 0.01 * model.l1.weight[0][0])
    #next_w = model.l1.weight[0][0] * 0.99 - w0_momentum

    #w0_momentum = next_m

    #print(model.l1.weight[0][0] - next_w, 'Prev', 
    #  model.l1.weight[0][0], 'next', next_w)

    #print('Grad:', w0_grad, 'Momentum:', w0_momentum, 'Prev w0:',
    #  model.l1.weight[0][0],
    #  'Next w0:',
    #  next_w)

        # model.l1.weight.grad) # .grad[0][0])


    lr1 = optimizer.param_groups[0]["lr"]
    print('Epoch:', epoch, 'Loss:', network.item(), 'Learning Rate:', lr1)

    optimizer.step()
    #scheduler.step()

    print('DIFF:', model.l1.weight[0][0].item() - next_w)
    obj_w = next_w

    # print('grad', model.l1.weight.grad[0][0], w0_grad)
    # print('Gradient of yhat_0 at noise 0:', yhat_00, yhat_00_grad, w0_grad, model.l1.weight.grad[0][0])
    optimizer.zero_grad()

    # model.l1.weight.data[0][0] = obj_w

    cost.append(network.item())

print(model.l1.state_dict())

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

# Final clipper to -1 or +1 only
binary_cmos_clipper() 

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
  y_hat = model(X[i])
  print(f'x0: { nn_to_py(x[0].item()) },\tx1: { nn_to_py(x[1].item()) },\ty_hat: { y_hat.item() }\ty_should: { Y[i].item() }')

