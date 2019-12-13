import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_y = [52, 52, 55, 55, 55, 56, 55, 57, 54, 58]
val_y = [41, 46, 43, 44, 44, 47, 44, 44, 46, 43]

font = {'family' : 'normal',
        'size'   : 25}

plt.rc('font', **font)
fig = plt.figure(figsize=(25,5))
ax = plt.axes()
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.yticks(np.arange(40, 60, 4.0))
ax.plot(x, train_y, label='Train')
ax.plot(x, val_y, label='Validation')
ax.set_xlabel('# of Epochs')
ax.set_ylabel('Loss (Percentage)')
ax.set_title('Epochs vs. Loss (MNIST)')
plt.tight_layout()
plt.legend()
plt.show()
