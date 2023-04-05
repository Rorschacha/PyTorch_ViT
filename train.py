import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm




# settings
device="cuda"
batch_size = 8
epochs = 20
save_after_epochs=20
lr = 3e-5
gamma = 0.7

#1.1
from models.vit import ViT
model = ViT(
    image_size = 224,
    patch_size = 8,
    num_classes = 2,
    dim = 1024,
    depth = 16,
    heads =16,
    mlp_dim = 4096,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)
# model.to(device)


#1.2 loss function
criterion = nn.CrossEntropyLoss()

#1.3 optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

#2.1 dataloader
from dataset import train_transforms,CatsDogsDataset,read_anno_pc
path_tr = r"D:\datasets\PracticeSets\cats and dogs\data\annotation_train.txt"
path_va = r"D:\datasets\PracticeSets\cats and dogs\data\annotation_valid.txt"

labels_tr = read_anno_pc(path_tr)
labels_va = read_anno_pc(path_va)
train_data=CatsDogsDataset(labels_tr, transform=train_transforms)
valid_data = CatsDogsDataset(labels_va, transform=train_transforms)
train_loader=DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
valid_loader=DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
def train():
    if True: # load path
        path_model=r"pre_model_epoch_40.pth"
        load_pretrained_model(load_model_path=path_model)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):

            #torch.cuda.empty_cache()
            input = data.to(device)
            #label_t=torch.tensor(label)
            label = label.to(device)


            #2.31
            output = model(input)
            #2.32
            loss = criterion(output, label)
            #2.33
            optimizer.zero_grad()
            #2.34
            loss.backward()

            #2.35
            optimizer.step()

            #acc = (output.argmax(dim=1) == label).float().mean()
            #out (batch_size,classes)
            predicted_class_indices = output.argmax(dim=1)
            correct_predictions = predicted_class_indices == label
            float_correct_predictions = correct_predictions.float() # True 1.0, False 0.0
            acc = float_correct_predictions.mean()

            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

            #print("loss",loss)
        print("Epoch : {} - loss : {:.4f} - acc: {:.4f} ".format(epoch,epoch_loss,epoch_accuracy))


        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print("val_loss : {:.4f} - val_acc: {:.4f}\n".format(epoch_val_loss,epoch_val_accuracy))

    if True:
        if (epoch + 1) % save_after_epochs == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")
            print(f"Model parameters saved at epoch {epoch + 1}")

    pass
#
#


#debug
def get_gpu_memory_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory

        print(f"Total GPU Memory: {total_memory / (1024 ** 2):.2f} MiB")
        print(f"Allocated Memory: {allocated_memory / (1024 ** 2):.2f} MiB")
        print(f"Reserved Memory: {reserved_memory / (1024 ** 2):.2f} MiB")
        print(f"Free Memory: {free_memory / (1024 ** 2):.2f} MiB")
    else:
        print("GPU is not available.")


def load_pretrained_model(load_model_path=None):
    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))
        print(f"Loaded model parameters from {load_model_path}")


def main():
    #get_gpu_memory_info()
    train()


if __name__ == '__main__':
    main()


