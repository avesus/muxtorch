import torch
import matplotlib.pyplot as plt
 
# generate synthetic the data
X = torch.arange(-30, 30, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] <= -10)] = 1.0
Y[(X[:, 0] > -10) & (X[:, 0] < 10)] = 0.5
Y[(X[:, 0] > 10)] = 0
 
#plt.plot(X, Y)
#plt.show()
 
# Define the class for single layer NN
class one_layer_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, hidden_neurons, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size) 
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None

    # prediction function
    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.nn.functional.sigmoid(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.nn.functional.sigmoid(self.linear_two(self.act))
        return y_pred
 
# create the model 
model = one_layer_net(1, 500, 1)  # 2 represents two neurons in one hidden layer
 
def criterion(y_pred, y):
    out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    return out

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
 
# Define the training loop
epochs=2000
cost = []
total=0
for epoch in range(epochs):
    total=0
    epoch = epoch + 1
    for x, y in zip(X, Y):
        yhat = model(x)
        loss = criterion(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # get total loss 
        total+=loss.item() 
    cost.append(total)

    '''
    if epoch % 1000 == 0:
        print(str(epoch)+ " " + "epochs done!") # visualze results after every 1000 epochs   
        # plot the result of function approximator
        plt.plot(X.numpy(), model(X).detach().numpy())
        plt.plot(X.numpy(), Y.numpy(), 'm')
        plt.xlabel('x')
        plt.show()
    '''
 
plt.plot(X.numpy(), model(X).detach().numpy())
plt.plot(X.numpy(), Y.numpy(), 'm')
plt.xlabel('x')
plt.show()

# plot the cost
plt.plot(cost)
plt.xlabel('epochs')
plt.title('cross entropy loss')
plt.show()

