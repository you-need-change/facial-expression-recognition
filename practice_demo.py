import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import warnings

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from scipy.spatial import distance

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import display, HTML, Video

from PIL import Image
import urllib.request


# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

train_folder = 'C:\\Users\\XIEMIAN\\Downloads\\Compressed\\archive\\train'
test_folder = 'C:\\Users\\XIEMIAN\\Downloads\\Compressed\\archive\\test'

# Function to load images from a folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion_folder in os.listdir(folder):
        label = emotion_folder
        for filename in os.listdir(os.path.join(folder, emotion_folder)):
            img = cv2.imread(os.path.join(folder, emotion_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

# Load images and labels from train and test folders
train_images, train_labels = load_images_from_folder(train_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Verify the shape of the datasets
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

from imblearn.over_sampling import SMOTE

# Identify the indices of "Disgust" and "Surprise" classes
disgust_indices = np.where(train_labels == "disgust")[0]
surprise_indices = np.where(train_labels == "surprise")[0]

# Combine the indices of both classes
indices_to_oversample = np.concatenate([disgust_indices, surprise_indices])

# Extract images and labels for the selected indices
X_to_oversample = train_images[indices_to_oversample]
y_to_oversample = train_labels[indices_to_oversample]

# Reshape the images to 2D (if needed)
X_to_oversample_2d = X_to_oversample.reshape(X_to_oversample.shape[0], -1)

# Define the oversampling strategy
oversample = SMOTE()

# Apply SMOTE to the selected classes
X_resampled, y_resampled = oversample.fit_resample(X_to_oversample_2d, y_to_oversample)

# Reshape the oversampled data back to its original shape
X_resampled = X_resampled.reshape(-1, *train_images.shape[1:])

# Check the new class distribution
unique, counts = np.unique(y_resampled, return_counts=True)
print(dict(zip(unique, counts)))

# Filter oversampled data for "disgust" class
disgust_resampled_indices = np.where(y_resampled == "disgust")[0]

# Define the maximum number of samples for each class
max_surprise_samples = 500
max_disgust_samples = 2000

# Identify the indices of the "surprise" and "disgust" classes in the resampled data
surprise_indices = np.where(y_resampled == "surprise")[0]
disgust_indices = np.where(y_resampled == "disgust")[0]

# Select the first 500 samples for "surprise" and the first 1500 samples for "disgust"
selected_surprise_indices = surprise_indices[:max_surprise_samples]
selected_disgust_indices = disgust_indices[:max_disgust_samples]

# Concatenate the original train images with the selected oversampled images
final_train_images = np.concatenate([train_images, X_resampled[selected_surprise_indices], X_resampled[selected_disgust_indices]], axis=0)

# Create labels for the selected oversampled images
selected_surprise_labels = np.full(len(selected_surprise_indices), "surprise")
selected_disgust_labels = np.full(len(selected_disgust_indices), "disgust")

# Concatenate the original train labels with the selected oversampled labels
final_train_labels = np.concatenate([train_labels, selected_surprise_labels, selected_disgust_labels], axis=0)

print('final_train_images.shape: ', final_train_images.shape, 'final_train_lables.shape: ', final_train_labels.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout3(x)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# training data
X_train_tensor = torch.tensor(final_train_images)
X_train_tensor = X_train_tensor.float()
X_train_tensor = torch.unsqueeze(X_train_tensor, 1)

print(X_train_tensor.shape, X_train_tensor.dtype)


#testing data
X_test_tensor = torch.tensor(test_images)
X_test_tensor = X_test_tensor.float()
X_test_tensor = torch.unsqueeze(X_test_tensor, 1)

print(X_test_tensor.shape, X_test_tensor.dtype)

#training data
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(final_train_labels)
y_train_tensor = torch.tensor(y_train_encoded)

class_names = label_encoder.classes_
print('class_names: ', class_names)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

#testing data
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(test_labels)
y_test_tensor = torch.tensor(y_test_encoded)

class_names = label_encoder.classes_
print('class_names: ', class_names)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Instantiate the model
num_classes = 7
model = CNN2(num_classes)
model.to(device)

# Define your loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

train_loss = []
train_acc = []
test_acc = []

def train(epoch):

    model.train()

    correct = 0
    running_loss = 0.0
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # 计算预测值
        pred = torch.max(outputs.data, dim=1)[1]

        # Calculate the loss
        loss = criterion(outputs, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (pred == target).sum().item()

    # 保存网络模型结构
    if epoch >= 46 and epoch % 2 == 0:
        torch.save(model.state_dict(), './model/fer_model_' + str(epoch) + '.pth')

    average_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / len(train_dataset)

    train_loss.append(average_loss)
    train_acc.append(accuracy)

    print("Epoch {}\nLoss {:.4f} Accuracy {}/{} ({:.0f}%)"
          .format(epoch+1, average_loss, correct, len(train_dataset),accuracy))

true_labels = []
predicted_labels = []

def test():
    #  Evaluate the model
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        # Iterate over the dataset or batches
        for inputs, labels in test_loader:
            # Forward pass

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Append true and predicted labels to lists
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Count total number of samples
            total += labels.size(0)

            # Count number of correct predictions
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_acc.append(accuracy)

        print('Accuray on test data is {:.2f}%'.format(accuracy))

for epoch in range(70):
    train(epoch)
    test()

torch.save(model.state_dict(), './model/fer_model_final.pth')

def print_plot1(train_plot, vaild_plot, train_text, vaild_text, ac, name, i):
    x = [i for i in range(1, len(train_plot) + 1)]
    plt.figure(i + 1)
    plt.plot(x, train_plot, label=train_text)
    plt.plot(x[-1], train_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (train_plot[-1]) if ac else "%.4f" % (train_plot[-1]), xy=(x[-1], train_plot[-1]))
    plt.plot(x, vaild_plot, label=vaild_text)
    plt.plot(x[-1], vaild_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (vaild_plot[-1]) if ac else "%.4f" % (vaild_plot[-1]), xy=(x[-1], vaild_plot[-1]))
    plt.legend()
    plt.savefig(name)

def print_plot(train_plot, train_text, ac, name, i):
    x = [i for i in range(1, len(train_plot) + 1)]
    plt.figure(i + 1)
    plt.plot(x, train_plot, label=train_text)
    plt.plot(x[-1], train_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (train_plot[-1]) if ac else "%.4f" % (train_plot[-1]), xy=(x[-1], train_plot[-1]))
    plt.legend()
    plt.savefig(name)

print_plot(train_loss,"train_loss",False,"loss.jpg", 1)
print_plot1(train_acc, test_acc,"train_acc","vaild_acc",True,"ac.jpg", 2)
