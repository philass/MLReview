import torch
import torchvision

batch_size_train = 64
batch_size_test = 1000

train_data = torchvision.datasets.MNIST('.', train=True, download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)

test_data = torchvision.datasets.MNIST('.', train=False, download=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=0)

network = Net()

# So things are reproducable
random_seed = 1
torch.manual_seed(random_seed)

# Paramaters for training
learning_rate = 0.01
momentum = 0.5

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()

    # Helpful logging
    if batch_idx % 100 == 0:
      print(f'Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item()}')


def test():
  network.eval()
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  print(f'Accuracy: {correct}/{len(test_loader.dataset)}')

n_epochs = 7

for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
