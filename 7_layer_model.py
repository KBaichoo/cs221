
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
from torch.optim import lr_scheduler


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 96, 3, 1, padding=1)
        self.conv6_bn = nn.BatchNorm2d(96)
        self.conv7 = nn.Conv2d(96, 96, 3, 1, padding=1)
        self.conv7_bn = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(98304, 3)

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)

        self.apply(init_weights)

    def forward(self, x):

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        # x = x.view(x.shape[0], 98304)
        # x = self.fc1(x)

        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = x.view(x.shape[0], 98304)
        x = self.fc1(x)
        return x




ce_loss = torch.nn.CrossEntropyLoss(size_average=False)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print('----------------------------------------------------------------')
        # print()
        # print(data.shape)
        # print()
        # print(target.shape)
        # print()
        # print(output.shape)
        # print()
        # print('----------------------------------------------------------------')
        loss = ce_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validation(args, model, device, validation_loader, best_loss_percentage, type):
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            # imshow(torchvision.utils.make_grid(data))
            data, target = data.to(device), target.to(device)
            output = model(data)


            # print('output: ', output)
            # print('target: ', target)
            # validation_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            validation_loss += ce_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss_total = validation_loss
    validation_loss /= len(validation_loader.dataset)

    if 100. * correct / len(validation_loader.dataset) > best_loss_percentage:
        best_loss_percentage = 100. * correct / len(validation_loader.dataset)

    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(type,
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

    return best_loss_percentage, validation_loss_total

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # imshow(torchvision.utils.make_grid(data))
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print('output: ', output)
            # print('target: ', target)
            # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss += ce_loss(output, target)
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

            # print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Binarize(object):
    """Applies Laplacian. Args - kernel size."""

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample):
        y = torch.zeros(sample.size())
        x = torch.ones(sample.size())
        return torch.where(sample > self.threshold, x, y)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--validation-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for validationing (default: )')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
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
                                 transforms.CenterCrop((128, 128)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                                 Binarize(0.2165)
                             ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./frames/validation/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.CenterCrop((128, 128)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                                 Binarize(0.2165)
                             ])),
        batch_size=args.validation_batch_size, shuffle=True, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./frames/test/',
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.CenterCrop((128, 128)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,)),
                                 Binarize(0.2165)
                             ])),
        batch_size=args.validation_batch_size, shuffle=True, drop_last=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    best_loss_percentage = -1
    # old_best_loss_percentage = -1

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        validation(args, model, device, train_loader, best_loss_percentage, 'train')
        best_loss_percentage, current_loss = validation(args, model, device, validation_loader, best_loss_percentage, 'validation')
        # if best_loss_percentage > old_best_loss_percentage:
        #     old_best_loss_percentage = best_loss_percentage
        #     if (args.save_model):
        #         torch.save(model.state_dict(),"resnet50.pt")
        scheduler.step(current_loss)

    print("Best validation loss: {}%".format(best_loss_percentage))

    if (args.save_model):
        torch.save(model.state_dict(),"7layer.pt")

    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
