from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np

# weights = [0] * 1000
# weights[0] = 0.23
# weights[1] = 0.30
# weights[2] = 0.47
# ce_loss = torch.nn.CrossEntropyLoss(size_average=False, weight=torch.FloatTensor(weights).cuda())

ce_loss = torch.nn.CrossEntropyLoss(size_average=False)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = ce_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def validation(args, model, device, validation_loader, best_loss_percentage):
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

    validation_loss /= len(validation_loader.dataset)

    if 100. * correct / len(validation_loader.dataset) > best_loss_percentage:
        best_loss_percentage = 100. * correct / len(validation_loader.dataset)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

    return best_loss_percentage

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

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--validation-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for validation (default: )')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: )')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('./frames/train/',
                       transform=transforms.Compose([
                           transforms.Grayscale(num_output_channels=3),
                           transforms.Resize((256,256)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    validation_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('./frames/validation/',
                       transform=transforms.Compose([
                           transforms.Grayscale(num_output_channels=3),
                           transforms.Resize((256,256)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.validation_batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('./frames/test/',
                       transform=transforms.Compose([
                           transforms.Grayscale(num_output_channels=3),
                           transforms.Resize((256,256)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss_percentage = -1
    # old_best_loss_percentage = -1

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        best_loss_percentage = validation(args, model, device, validation_loader, best_loss_percentage)
        # if best_loss_percentage > old_best_loss_percentage:
        #     old_best_loss_percentage = best_loss_percentage
        #     if (args.save_model):
        #         torch.save(model.state_dict(),"resnet50.pt")

    print("Best validation loss: {}%".format(best_loss_percentage))

    if (args.save_model):
        torch.save(model.state_dict(),"resnet50.pt")

    test(args, model, device, test_loader)


if __name__ == '__main__':
    main()
