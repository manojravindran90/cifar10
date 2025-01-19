import os
import time
import torch
import tarfile
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
# import matplotlib
# import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, steps_per_epoch=len(train_data), epochs=num_epochs
)
    history = []
    total_items = 0
    total_time = 0
    # Traning the model
    for epoch in range(epochs):
        epoch_start_time = time.time() 
        model.train()
        for batch in train_data:
            start_time = time.time()
            train_loss = []
            # Use autocast with the appropriate device type
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    loss = model.training_step(batch)
                    train_loss.append(loss)
            else:
                loss = model.training_step(batch)
                train_loss.append(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            end_time = time.time()
            batch_time = end_time - start_time
            batch_size = len(batch[0])  # Assuming batch[0] contains input images
            total_items += batch_size
            total_time += batch_time
             # Calculate throughput
        epoch_end_time = time.time()
        items_per_second = total_items / total_time if total_time > 0 else 0
        avg_train_loss = torch.tensor(train_loss).mean().item()
        # Validation phase
        result = eval(model, val_data)
        result['train_loss'] = avg_train_loss
        result['items_per_second'] = items_per_second
        result['epoch_time'] = epoch_end_time - epoch_start_time
        print(  f"Epoch {epoch + 1}/{epochs}, "
                f"Train Loss: {result['train_loss']:.4f}, "
                f"Val Loss: {result['val_loss']:.4f}, "
                f"Val Accuracy: {result['val_acc']:.4f}, "
                f"Throughput: {items_per_second:.2f} items/second",
                f"Time spent: {result['epoch_time']:2f}"
         )

        history.append(result)
    return history

if __name__ == "__main__":

    random_seed = 42
    torch.manual_seed(random_seed)

    # Download image dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, './')
    data_dir = './data/cifar10'

    # Check if the data directory already exists
    if not os.path.exists(data_dir):
        print("Extracting the archive...")
        with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='./data')
        print("Extraction complete.")
    else:
        print("Data folder already exists, skipping extraction process")

    # create dataset
    dataset_from_train_folder = ImageFolder(data_dir + '/train', transform=ToTensor())
    img, label = dataset_from_train_folder[0]

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # train/val split
    val_size = 5000
    batch_size = 512
    train_size = len(dataset_from_train_folder) - val_size
    train_split, val_split = random_split(dataset_from_train_folder, [train_size, val_size])

    train_data = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_data = DataLoader(val_split, batch_size=batch_size * 2, num_workers=12, pin_memory=True)


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

                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 128, 256, 8, 8
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 128, 256, 8, 8
                nn.BatchNorm2d(512),
                nn.ReLU(),
                # nn.MaxPool2d(2,2), # 128, 256, 4, 4

                nn.Flatten(),
                nn.Linear(512 * 4 * 4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
        
        def forward(self, x):
            return self.network(x)

    class Plain34LayerCifar10(ImageClassificationBase):
        def __init__(self, num_classes=10):
            super(Plain34LayerCifar10, self).__init__()

            # Initial Convolution + BatchNorm + ReLU
            self.initial = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )

            # Repeating Layers
            self.layer1 = self._make_layer(64, 64, 3)  # 3 residual blocks * 2 Conv per block = 6 layers
            self.layer2 = self._make_layer(64, 128, 4, stride=2)  # 4 blocks * 2 Conv per block = 8 layers
            self.layer3 = self._make_layer(128, 256, 6, stride=2)  # 6 blocks * 2 Conv per block = 12 layers
            self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 3 blocks * 2 Conv per block = 6 layers

            # Fully Connected Layer
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = []
            layers.append(self._conv_block(in_channels, out_channels, stride))  # First block
            for _ in range(1, blocks):
                layers.append(self._conv_block(out_channels, out_channels))  # Remaining blocks
            return nn.Sequential(*layers)

        def _conv_block(self, in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.initial(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    model = Plain34LayerCifar10().to(device)

    num_epochs = 50
    opt_func = torch.optim.AdamW
    lr = 0.01
    history = fit(epochs=num_epochs, lr=lr, train_data=train_data, val_data=val_data, model=model, opt_func=opt_func)

# Pre Batchnorm
# end of epcoch 9: train loss: 0.38342195749282837, val_loss: 0.8656366467475891, val_acc: 0.7635455131530762

# Post Batchnorm 
# end of epcoch 9: train loss: 0.24247115850448608, val_loss: 0.6352024078369141, val_acc: 0.8247932195663452