
# BASED ON: https://github.com/pytorch/examples/tree/master/mnist


from __future__ import print_function
import argparse
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(1024, 1000)
        # We simply have 3 classification classes...
        self.fc2 = nn.Linear(1000, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def imshow(img):
    img = img * 255
    # img = (img * 0.3081) + 0.1307     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(args, model, device, train_loader, optimizer, epoch):
    # get some random training images
    dataiter = iter(train_loader)

    """
    for i in range(50):
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        """

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # TODO(kbaichoo): add a regulazation term
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def validation(args, model, device, validation_loader):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            validation_loss += F.nll_loss(output,
                                          target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)

    print('\nvalidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


class Binarize(object):
    """Applies Laplacian. Args - kernel size."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample):
        y = torch.zeros(sample.size())
        x = torch.ones(sample.size())
        print('mean:{}'.format(torch.mean(sample)))
        return torch.where(sample > self.threshold, x, y)

# TODO(kbaichoo): add more layers....


class Laplace(object):
    """Applies Laplacian. Args - kernel size."""

    def __init__(self, ksize):
        self.ksize = ksize
        self.laplace = kornia.filters.Laplacian(ksize)

    def __call__(self, sample):
        img = torch.unsqueeze(sample, dim=0)
        return torch.squeeze(self.laplace(img), dim=0)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--validation-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for validationing (default: )')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('using cuda')
    else:
        print('no cuda...')

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

    # TODO(kbaichoo): should remove random rotation from test + from
    # training (b/c exhaustive); further use validation.
    # TODO(kbaichoo): add back normalization / fix the imshow func.
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./frames/train/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize((64, 64)),
                                 transforms.ToTensor()
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./frames/validation/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize((64, 64)),
                                 transforms.ToTensor()
                             ])),
        batch_size=args.validation_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./frames/test/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize((64, 64)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                                 Binarize(0.2165)
                             ])),
        batch_size=args.validation_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        validation(args, model, device, validation_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()