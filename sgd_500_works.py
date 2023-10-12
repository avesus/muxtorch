import torch
import numpy as np

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

class one_layer_net (torch.nn.Module):    

    def __init__ (self):
        super(one_layer_net, self).__init__()


        self.l1 = torch.nn.Linear(1, 500)
        self.l2 = torch.nn.Linear(500, 1)

        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)

        '''
        torch.nn.init.constant_(self.l2.bias[0], 1.0)
        torch.nn.init.constant_(self.l2.weight[0][0], 0.0)

        torch.nn.init.constant_(self.l1.weight[0][0], 0.0)
        torch.nn.init.constant_(self.l1.bias[0], 0.0)

        torch.nn.init.constant_(self.l2.weight[0][1], -1.0)
        torch.nn.init.constant_(self.l1.weight[1][0], 0.3)
        torch.nn.init.constant_(self.l1.bias[1], 6.4)

        torch.nn.init.constant_(self.l2.weight[0][2], 0.5)
        torch.nn.init.constant_(self.l1.weight[2][0], 0.6)
        torch.nn.init.constant_(self.l1.bias[2], 11.8)

        torch.nn.init.constant_(self.l2.weight[0][3], -0.6)
        torch.nn.init.constant_(self.l1.weight[3][0], 1.0)
        torch.nn.init.constant_(self.l1.bias[3], -20.0)

        torch.nn.init.constant_(self.l2.weight[0][4], 0.1734)
        torch.nn.init.constant_(self.l1.weight[4][0], 3.46)
        torch.nn.init.constant_(self.l1.bias[4], -72.0)
        '''

        # self.l3 = torch.nn.Linear(3, 1) 
        '''
        torch.nn.init.normal_(self.l1.weight, mean=0, std=0.1/4.0)
        torch.nn.init.normal_(self.l2.weight, mean=0, std=0.1)
        #torch.nn.init.normal_(self.l3.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.l1.bias, mean=0, std=0.001/4.0)
        torch.nn.init.normal_(self.l2.bias, mean=0, std=0.001)
        #torch.nn.init.normal_(self.l3.bias, mean=0, std=0.001)
        
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)
        torch.nn.init.xavier_uniform_(self.l3.weight)
        '''

        print('w00, w01', self.l1.weight[0][0], self.l1.weight[1][0])
        print('b00, b01', self.l1.bias[0], self.l1.bias[1])

        print('w10, w11', self.l2.weight[0][0], self.l2.weight[0][1])
        print('b10', self.l2.bias[0])

        print(self.l1.state_dict())
        print(self.l2.state_dict())
        # print(self.l3.state_dict())

    def forward (self, x) :
        '''
        manual = 0.5 + self.l2.bias[0] + self.l2.weight[0][0] * ReLU(
             self.l1.weight[0][0] * x * 1.0/30.0 + self.l1.bias[0]
           ) + self.l2.weight[0][1] * ReLU(
             self.l1.weight[1][0] * x * 1.0/30.0 + self.l1.bias[1]
           ) + self.l2.weight[0][2] * ReLU(
             self.l1.weight[2][0] * x * 1.0/30.0 + self.l1.bias[2]
           ) + self.l2.weight[0][3] * ReLU(
             self.l1.weight[3][0] * x * 1.0/30.0 + self.l1.bias[3]
           ) + self.l2.weight[0][4] * ReLU(
             self.l1.weight[4][0] * x * 1.0/30.0 + self.l1.bias[4]
           )
        '''
        torch_tensors = torch.nn.functional.relu(
          self.l2(
                    torch.nn.functional.relu(
                        self.l1(x)
                    )
          )
        )
        # print(manual - torch_tensors)

        return torch_tensors
 
# create the model 
model = one_layer_net()
 

# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
 
# Define the training loop
epochs = 2000
cost = []
total = 0

training_data = list(zip(X, Y))


for epoch in range(epochs):
    total = 0
    epoch = epoch + 1

    network = 0.0

    # for i in range(len(training_data)) :
    for x, y in zip(X, Y) :
    #for i in np.random.permutation(range(len(training_data))) :
        #x, y = training_data[i]
        # y = y - 0.5
        #x = x + 0.002 * np.random.normal()
        #y = y + 0.002 * np.random.normal()
        # x = x * 1.0 / 30.0

        # print(x, y)

        # forward
        yhat = model(x)

        #network = network + -1.0 * torch.mean(y * torch.log(yhat) + (1.0 - y) * torch.log(1.0 - yhat))
        #network = network + -1.0 * (y * torch.log(yhat) + (1.0 - y) * torch.log(1.0 - yhat))
          
        network = network + torch.square(torch.sub(y, yhat))
        # network = torch.square(torch.sub(y, yhat))
        #network = network + torch.nn.MSELoss()(y, yhat)

    # network = network / 120.0
    # backward
    network.backward()



    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.1)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()
    # get total loss 
    total += network.item() 

    # print(total)

    cost.append(total)
    # if epoch % 500 == 0:

print(model.l1.state_dict())
print(model.l2.state_dict())

# print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs   
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

