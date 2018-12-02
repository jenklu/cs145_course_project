from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import pandas as pd
import get_data
import numpy as np
import bare_logging as lg
#from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(11, 33)
        self.fc2 = nn.Linear(33, 19)
        self.fc3 = nn.Linear(19,5)
        self.act_function = nn.PReLU()

    def forward(self, x):
        x = self.act_function(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.act_function(self.fc2(x))
        x = self.fc3(x)
        return nn.Softmax(dim=1)(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    for batch_idx, batch in enumerate(train_loader):
        data = batch['features']
        target = batch['expect'].max(1)[1]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        #loss = torch.sqrt(loss) #RMSE loss function
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            #print("TEST: {} {}".format(output[0:5].max(1)[1],target[0:5]))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

def test(args, model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch['features']
            target = batch['expect'].float()
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1)[1].float()
            test_loss += ((pred - target) ** 2).sum() # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss = np.sqrt(test_loss)
    printstr = 'Epoch {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    print(printstr)
    print("Sample - Predict:{}, Expected: {}".format(pred[0:4], target[0:4]))
    lg.log(printstr, 0)

def main():
    # Training settings
    lg.fileSetup()
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=3200, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(use_cuda)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print( torch.cuda.device_count())
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    #transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                    download=True, transform=transform)

    training_data = get_data.YelpTrainingDataset()
    validate_data = get_data.YelpTestingDataset()
    train_loader = torch.utils.data.DataLoader(training_data,batch_size = args.batch_size, shuffle = True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(validate_data, batch_size = args.test_batch_size, shuffle = True, num_workers=2)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader,epoch)


if __name__ == '__main__':
    main()