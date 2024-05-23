import urllib.request
import cv2

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import seaborn as sns
from statistics import mode


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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model = CNN2(7)
model.load_state_dict(torch.load('./model/fer_model_52.pth'))
model.to(device)
#
class_names = ['angry', 'diguest', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# def process_live_video(face_Cascade):
#
#     print("Model has been loaded")
#
#     frame_window = 10
#
#     # class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
#
#     emotion_window = []
#
#     cap = cv2.VideoCapture(0)
#
#     if not cap.isOpened():
#         print("Error: Failed to open camera.")
#         return
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break
#
#         cap.set(cv2.CAP_PROP_AUDIO_POS, 0.3)
#         # Convert frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
#
#         for (x, y, w, h) in faces:
#             # Crop face region
#
#             print('x:', x, 'y:', y, 'w:', w, 'h:', h)
#             face_img = frame[y:y + h, x:x + w]
#
#             # Convert face_img to PIL image
#             face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#             face_img_resized = face_img_pil.resize((48, 48))
#             face_img_gray = face_img_resized.convert('L')
#
#             # Convert PIL image to NumPy array
#             face_img_np = np.array(face_img_gray)
#             face_img_tensor = torch.tensor(face_img_np, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(
#                 'cuda' if torch.cuda.is_available() else 'cpu')
#
#             # Send face image tensor to the model
#             with torch.no_grad():
#                 outputs = model(face_img_tensor)
#
#             # Get predicted emotion label
#             _, predicted = torch.max(outputs, 1)
#             predicted_label = class_names[predicted.item()]
#
#             # Draw rectangle around detected face and display predicted emotion label
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#             emotion_window.append(predicted_label)
#
#             if len(emotion_window) >= frame_window:
#                 emotion_window.pop(0)
#
#             try:
#                 # 获得出现次数最多的分类
#                 emotion_mode = mode(emotion_window)
#             except:
#                 continue
#
#             # Draw the main text in orange
#             cv2.putText(frame, emotion_mode, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)
#
#             # Draw slightly offset text to create a bold effect in orange
#             cv2.putText(frame, emotion_mode, (x + 1, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)
#             cv2.putText(frame, emotion_mode, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)
#
#         print(frame.shape)
#
#         cv2.imshow('FRAME', frame)
#
#         key = cv2.waitKey(1)
#         if key == 27:  # exit on ESC
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

def img_recognition(face_casecade, picture):

    img = cv2.imread(picture)

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop face region
        print('x:', x, 'y:', y, 'w:', w, 'h:', h)
        face_img = img[y:y + h, x:x + w]

        # Convert face_img to PIL image
        face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_img_resized = face_img_pil.resize((48, 48))
        face_img_gray = face_img_resized.convert('L')

        # Convert PIL image to NumPy array
        face_img_np = np.array(face_img_gray)
        face_img_tensor = torch.tensor(face_img_np, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Send face image tensor to the model
        with torch.no_grad():
            outputs = model(face_img_tensor)

        # Get predicted emotion label
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]

        # Draw rectangle around detected face and display predicted emotion label
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw the main text in orange
        cv2.putText(img, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 2)

        # Draw slightly offset text to create a bold effect in orange
        cv2.putText(img, predicted_label, (x + 1, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 2)
        cv2.putText(img, predicted_label, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 2)

        new_img = picture_path + 'bq_' + img1

        cv2.imwrite(new_img, img)


# test_folder = 'C:\\Users\\XIEMIAN\\Downloads\\Compressed\\archive\\test'
#
# # Function to load images from a folder
# def load_images_from_folder(folder):
#     images = []
#     labels = []
#     for emotion_folder in os.listdir(folder):
#         label = emotion_folder
#         for filename in os.listdir(os.path.join(folder, emotion_folder)):
#             img = cv2.imread(os.path.join(folder, emotion_folder, filename), cv2.IMREAD_GRAYSCALE)
#             if img is not None:
#                 images.append(img)
#                 labels.append(label)
#     return images, labels
#
# # Load images and labels from train and test folders
# test_images, test_labels = load_images_from_folder(test_folder)
#
# # Convert lists to numpy arrays
# test_images = np.array(test_images)
# test_labels = np.array(test_labels)
#
# # Verify the shape of the datasets
# print("Test images shape:", test_images.shape)
# print("Test labels shape:", test_labels.shape)
#
# #testing data
# X_test_tensor = torch.tensor(test_images)
# X_test_tensor = X_test_tensor.float()
# X_test_tensor = torch.unsqueeze(X_test_tensor, 1)
#
# print('X_test_tensor.shape: ',X_test_tensor.shape, 'X_test_tensor.dtype: ', X_test_tensor.dtype)
#
# #testing data
# label_encoder = LabelEncoder()
# y_test_encoded = label_encoder.fit_transform(test_labels)
# y_test_tensor = torch.tensor(y_test_encoded)
#
# class_names = label_encoder.classes_
# print(class_names)
#
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
#
# true_labels, predicted_labels = [], []
#
# def test():
#     #  Evaluate the model
#     model.eval()
#
#     correct, total = 0, 0
#
#     with torch.no_grad():
#         # Iterate over the dataset or batches
#         for inputs, labels in test_loader:
#             # Forward pass
#
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             outputs = model(inputs)
#
#             # Get predicted labels
#             _, predicted = torch.max(outputs.data, 1)
#
#             # Append true and predicted labels to lists
#             true_labels.extend(labels.cpu().numpy())
#             predicted_labels.extend(predicted.cpu().numpy())
#
#             # Count total number of samples
#             total += labels.size(0)
#
#             # Count number of correct predictions
#             correct += (predicted == labels).sum().item()
#
#         accuracy = 100 * correct / total
#         # test_acc.append(accuracy)
#
#         print('Accuray on test data is {:.2f}%'.format(accuracy))
#
# test()
#
# # Get the original class names
# class_names = label_encoder.classes_
#
# # Generate confusion matrix
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
# # Plot confusion matrix with class names
# plt.figure(figsize=(12, 6))
# sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False,
#             xticklabels=class_names, yticklabels=class_names)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.show()

# Function to load images from a folder
def load_images_from_folder(picture_path):
    images = []
    for filename in os.listdir(picture_path):
        if filename is not None:
            images.append(filename)
    return images

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# process_live_video(face_cascade)

picture_path = './img/'

pictures = load_images_from_folder(picture_path)

for img1 in pictures:
    picture = picture_path + img1
    img_recognition(face_cascade, picture)

#     print(picture)

    # img = cv2.imread(picture)
    #
    # new_picture = picture_path + 'n' + img1
    # cv2.imwrite(new_picture, img)