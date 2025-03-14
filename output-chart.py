import numpy as np
import matplotlib.pyplot as plt

# Data
with open('output.txt') as f:
    data = f.readlines()
    
data = [x.strip() for x in data]
data = [x if '-' in x else None for x in data]
data = list(filter(None, data))

train = []
val = []

for i in range(len(data)):
    if 'Train' in data[i]:
        train.append(i)
    if 'Validation' in data[i]:
        val.append(i)

train = [x.split('Train Loss: ')[1] for x in data if 'Train' in x]
train = [float(x) for x in train]

val = [x.split('Validation Loss: ')[1] for x in data if 'Validation' in x]
val = [float(x) for x in val]

x = np.arange(0, len(train))

# Plot
plt.plot(train, label='Train Loss', color='tab:blue')
plt.plot(val, label='Validation Loss', color='tab:red')
plt.scatter(x, train, color='tab:blue')
plt.scatter(x, val, color='tab:red')
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Validation Loss vs. Epoch')
plt.show()
