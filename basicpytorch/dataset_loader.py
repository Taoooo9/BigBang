import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter = ',', dtype = np.float32)
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]



dataset = DiabetesDataset()
dataloader = DataLoader(dataset = dataset, batch_size = 32,
                        shuffle = True)

for epoch in range(2):
    for index, data in enumerate(dataloader):
        x_data, y_data = data

        x_data, y_data = Variable(x_data), Variable(y_data)

        print(epoch, index, 'x:', x_data.data, 'y:', y_data.data)