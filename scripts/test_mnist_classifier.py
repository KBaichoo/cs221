#!/usr/local/bin/python3
# Simple program runs the mnist classifer model on a given image.
import argparse
import logging
import torch
from model import Net, Binarize
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file',
                        help='path to the model', required=True)
    parser.add_argument('--img',
                        help='path to the img', required=True)

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)

    # Pick device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print('using cuda')
    else:
        print('no cuda...')

    # Set up the model
    model = Net()
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()

    loader = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Binarize(0.2165)
    ])

    image = Image.open(args.img)
    image = loader(image).float()
    image.unsqueeze_(0)

    output = model(image)
    # get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)
    mapping = ['left', 'right', 'stay']
    print('Prediction:', mapping[pred.item()])
