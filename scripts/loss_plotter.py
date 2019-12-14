#!/usr/local/bin/python3
# Parses output from net to plot the epoch loss and testset loss
import matplotlib.pyplot as plt
import re


training_line = []
test_line = []
with open('resnet_results.txt') as f:
    line = f.readline()
    while line != '':
        if 'Train Epoch' in line:
            training_line.append(line)
        if 'Test set' in line:
            test_line.append(line)
        line = f.readline()


test_acc = []
for i in range(0, len(test_line)):
    line = test_line[i]
    results = re.search('(\(\d+%\))', line)
    if results is None:
        continue
    acc_str = results.group(1)
    print(acc_str)
    acc_int = float(acc_str.split('%')[0][1:])
    test_acc.append(acc_int)

# Get every 20th test line
epochs = [x for x in range(1, 501)]


loss_epoch_end = []
for i in range(0, len(training_line)):
    line = training_line[i]
    if i % 2 == 0:
        continue
    results = re.search('Loss: ([\d\.]+)', line)
    if results is None:
        continue
    loss_epoch = float(results.group(1))
    loss_epoch_end.append(loss_epoch)


plt.plot(epochs, test_acc[0:500], 'r--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.yscale('log')
plt.title('Accuracy over Epochs')
plt.show()
