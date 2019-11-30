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

ce_loss = torch.nn.CrossEntropyLoss(size_average=False)



def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: )')
    parser.add_argument('--validation-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for validation (default: )')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: )')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
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
    use_cuda = True

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder('./frames/test/',
                       transform=transforms.Compose([
                           transforms.Grayscale(num_output_channels=3),
                           transforms.Resize((256,256)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    classes = ('left', 'right', 'stay')

    device = torch.device("cuda")
    model = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=False).to(device)
    model.load_state_dict(torch.load('./resnet50.pt'))
    model.eval()

    # left_total = 0
    # left_correct = 0
    # right_total = 0
    # right_correct = 0
    # stay_total = 0
    # stay_correct = 0

    def test(args, model, device, test_loader, classes):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # imshow(torchvision.utils.make_grid(data))
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += ce_loss(output, target)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability


                _, predicted = torch.max(output, 1)

                # if classes[target[0]] == 'left':
                #     left_total += 1
                #     if classes[predicted[0]] == 'left':
                #         left_correct += 1
                #
                # if classes[target[0]] == 'right':
                #     right_total += 1
                #     if classes[predicted[0]] == 'right':
                #         right_correct += 1
                #
                # if classes[target[0]] == 'stay':
                #     stay_total += 1
                #     if classes[predicted[0]] == 'stay':
                #         stay_correct += 1

                if classes[target[0]] != classes[predicted[0]]:
                    print('GroundTruth: ', ' '.join('%5s' % classes[target[j]] for j in range(1)))
                    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                      for j in range(1)))
                    imshow(torchvision.utils.make_grid(data))
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test(args, model, device, test_loader, classes)

    print('left accuracy: ', left_correct/left_total)
    print('right accuracy: ', right_correct/right_total)
    print('stay accuracy: ', stay_correct/stay_total)

if __name__ == '__main__':
    main()
