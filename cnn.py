import os
import torch
import tarfile
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt

random_seed = 42
torch.manual_seed(random_seed)

# # Download image dataset
# dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
# download_url(dataset_url, './')

# # Extract from archive
# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#     tar.extractall(path='./data')

data_dir = './data/cifar10'
classes = os.listdir(data_dir + '/train')
print(classes)

bird_files = os.listdir(data_dir + '/train/bird')
print(f'len of items in bird classes in train dataset: {len(bird_files)}')
print(bird_files[:5])

# create dataset
dataset_from_train_folder = ImageFolder(data_dir + '/train', transform=ToTensor())
print(f'length of the dataset {len(dataset_from_train_folder)}')
img, label = dataset_from_train_folder[0]
print(img.shape, label)
print(f"list of classes: {dataset_from_train_folder.classes}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def show_example(img, label):
    print(f"Label: {dataset_from_train_folder.classes[label]}")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

# show_example(*dataset_from_train_folder[0])

# train/val split
val_size = 5000
batch_size = 128
train_size = len(dataset_from_train_folder) - val_size
train_split, val_split = random_split(dataset_from_train_folder, [train_size, val_size])

train_data = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_data = DataLoader(val_split, batch_size=batch_size*2, num_workers=0, pin_memory=True)

simple_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2,2)
)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        image, label = batch
        image, label = image.to(device), label.to(device)
        out = self(image)
        loss = F.cross_entropy(out, label)
        return loss
    
    def validation_step(self, batch):
        image, label = batch
        image, label = image.to(device), label.to(device)
        out = self(image)
        val_loss = F.cross_entropy(out, label)
        val_acc = accuracy(out, label)
        return {'val_loss':val_loss, 'val_acc': val_acc}

def accuracy(out, label):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == label).item() / len(preds))

class Cifar10CnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # expects input of 128, 3, 32, 32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1), # 128, 32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 128, 64, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 128, 64, 16, 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128, 128, 16, 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # 128, 128, 16, 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 128, 128, 8, 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 128, 256, 8, 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 128, 256, 8, 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 128, 256, 4, 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.network(x)

model = Cifar10CnnModel().to(device)


@torch.no_grad()
def eval(model, val_data):
    model.eval()
    val_loss_final = []
    val_acc_final = []
    out = [model.validation_step(batch) for batch in val_data]
    for item in out:
        val_loss_final.append(item['val_loss'])
        val_acc_final.append(item['val_acc'])
    # Compute mean for loss and accuracy
    mean_val_loss = torch.tensor(val_loss_final).mean().item()
    mean_val_acc = torch.tensor(val_acc_final).mean().item()
    return {'val_loss': mean_val_loss, 'val_acc': mean_val_acc}

def fit(epochs, lr, train_data, val_data, model, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []
    # Traning the model
    for epoch in range(epochs):
        model.train()
        for batch in train_data:
            train_loss = []
            loss = model.training_step(batch)
            train_loss.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_train_loss = torch.tensor(train_loss).mean().item()
        # Validation phase
        result = eval(model, val_data)
        result['train_loss'] = avg_train_loss
        print(f"end of epcoch {epoch}: train loss: {avg_train_loss}, val_loss: {result['val_loss']}, val_acc: {result['val_acc']}")
        history.append(result)
    return history

num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001
history = fit(epochs=num_epochs, lr=lr, train_data=train_data, val_data=val_data, model=model, opt_func=opt_func)

# Pre Batchnorm
# end of epcoch 9: train loss: 0.38342195749282837, val_loss: 0.8656366467475891, val_acc: 0.7635455131530762

# Post Batchnorm 
# end of epcoch 9: train loss: 0.24247115850448608, val_loss: 0.6352024078369141, val_acc: 0.8247932195663452