import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

name = 'ResNet18 with L2 Regularization, Batch Size 32'
filepath = 'final_paper_resnet18_l2.txt'
train_y = []
val_y = []
x = []
for i in range(1,151):
    x.append(i)

with open(filepath) as fp:
   line = fp.readline()
   while line:
       line = line.strip()
       if line.find('train set: ') != -1:
           # print(line[-4:-2])
           value = int(line[-4:-2])
           value = 100 if value == 0 else value
           train_y.append(value)
       if line.find('validation set: ') != -1:
           # print(line[-4:-2])
           val_y.append(int(line[-4:-2]))
       line = fp.readline()

font = {'family' : 'normal',
        'size'   : 25}
plt.rc('font', **font)
fig = plt.figure(figsize=(20,20))
ax = plt.axes()
plt.xticks(np.arange(min(x) - 1, max(x) + 10, 10.0))
plt.yticks(np.arange(0, 100, 5.0))
ax.plot(x, train_y, label='Train')
ax.plot(x, val_y, label='Validation')
ax.set_xlabel('Number of Epochs')
ax.set_ylabel('Percentage of Set Correct')
ax.set_title('Epochs vs. Percentage Correct ({})'.format(name) )
# plt.tight_layout()
plt.legend()
plt.show()
