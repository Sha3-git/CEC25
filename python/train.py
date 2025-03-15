import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import random
from torch.cuda.amp import autocast, GradScaler

NUM_IMAGES = 500  # number of images for training
MODEL_NAME = "final_model.pth"  # name of model

# transformation function 
transform = transforms.Compose([
    # resize for EfficientNet required input
    transforms.Resize((224, 224)),
    # covert greyscale to RGB as its best for EfficientNet
    transforms.Grayscale(num_output_channels=3),
    # do a horizontal flip to reduce overfitting
    transforms.RandomHorizontalFlip(),
    # do a rotation as MRIs can be on angles at times
    transforms.RandomRotation(15),
    # change it to a tensor for pytorch (multi-dem array)
    transforms.ToTensor(),
    # adjust pixels to fit RGB pattern (should increase convergance)
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# image folder as per standard
class CustomImageFolder(Dataset):
    # setup the class with the params
    def __init__(self, root_dir, transform=None, num_samples=NUM_IMAGES):
        self.root_dir = root_dir
        self.transform = transform
        self.num_samples = num_samples
        self.samples = self._get_samples()

    # a get samples call
    def _get_samples(self):
        samples = []
        # we use two folder names
        for folder_name in ["yes", "no"]:
            # get the path
            folder_path = os.path.join(self.root_dir, folder_name)
            # as long as it exists
            if os.path.isdir(folder_path):
                # grab some files
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and "test" not in f]
                random.shuffle(files) 
                # we want half from each folder
                files = files[:self.num_samples // 2] 
                # label them accordingly
                for file_path in files:
                    label = 1 if folder_name == "yes" else 0
                    samples.append((file_path, label))
        return samples

    def __len__(self):
        # get length
        return len(self.samples)

    def __getitem__(self, idx):
        # get an image item
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# the directory we using
data_dir = "CEC_2025"
# getting dataset with transforms and amount
dataset = CustomImageFolder(root_dir=data_dir, transform=transform, num_samples=NUM_IMAGES) 
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)

# create the model object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# we will use efficient net v2 - small (light)
model = models.efficientnet_v2_s(weights="IMAGENET1K_V1") 
# we want a binary yes or no classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
# move to cpu or gpu
model.to(device)

# cross entropy expects labels as 0,1 (standard)
# this is used for classifcation problems often
criterion = nn.CrossEntropyLoss() 
# update parameters in training for a loss rate of 0.001
# learning rate of 0.001 is often chosen as a good starting point for Adam optimizer.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# precision scaler setup improves performances and prevents underflow during 
# backpropagation (adjust weights moving)
scaler = GradScaler()

# good starting point for efficient net v2
epochs = 5 

for epoch in range(epochs):
    # train the model
    model.train()
    # running loss
    running_loss = 0.0
    # counter for prints
    image_counter = 0 
    # number of batches
    total_batches = len(train_loader)

    # for each batch with its images and labels
    for batch_idx, (images, labels) in enumerate(train_loader):
        # track batch number
        print(f"batch {batch_idx + 1}/{total_batches}...") 

        # def images
        images, labels = images.to(device), labels.to(device)

        # clear the gradient
        optimizer.zero_grad()

        # enable autocasting
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # scales the loss to prevent underflow
        # updates the model weights with the optimizer
        # adjusts the scaling factor for the next
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # add running loss
        running_loss += loss.item()

        # batch size
        batch_size = len(images)
        for i in range(batch_size):
            # just output image count; not the best since they batched
            image_counter += 1
            print(f"image {image_counter}/{len(train_loader.dataset)}")

    # print epoch current
    print(f"epoch {epoch + 1}/{epochs}, loss: {running_loss / len(train_loader)}")


torch.save(model.state_dict(), MODEL_NAME)

print(f"model saved as {MODEL_NAME}")
