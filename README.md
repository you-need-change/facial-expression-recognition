# 人脸表情识别
## practice_demo.py中的步骤
### 1、数据集加载
```
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
```
输出
```
Train images shape: (28709, 48, 48)
Train labels shape:`Train labels shape: (28709,)
Test`Test images shape: (7178, 48, 48)
Test labels shape: (7178,)
```
### 2、训练集数据过采样
```
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
```
最终训练集图片
![newplot (1)](https://github.com/you-need-change/facial-expression-recognition/assets/90135052/12f9a6f2-b47b-450f-9c23-41c04b40fb10)
输出
```
final_train_images.shape:  (31209, 48, 48) final_train_lables.shape:  (31209,)
```
### 3、创建模型和DataLoader
```
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
```
表情类别
> class_names:  ['angry' 'disgust' 'fear' 'happy' 'neutral' 'sad' 'surprise']
### 4、模型训练
```
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
```
### 5、模型测试
```
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
```
### 6、训练图像绘制
```
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
```
![loss](https://github.com/you-need-change/facial-expression-recognition/assets/90135052/4c613e3a-dee1-44ab-a804-a45baebddd92)

![ac](https://github.com/you-need-change/facial-expression-recognition/assets/90135052/542ed2bd-9f82-48f8-8487-1bddca3667e8)
## run_me.py中的步骤
### 1、测试模型
```
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
```
输出
![识别矩阵](https://github.com/you-need-change/facial-expression-recognition/assets/90135052/cf436a4b-9d33-4647-bb5f-de58811883bb)
### 2、摄像头测试
```
def process_live_video(face_Cascade):

    print("Model has been loaded")

    frame_window = 10

    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    emotion_window = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cap.set(cv2.CAP_PROP_AUDIO_POS, 0.3)
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Crop face region

            print('x:', x, 'y:', y, 'w:', w, 'h:', h)
            face_img = frame[y:y + h, x:x + w]

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            emotion_window.append(predicted_label)

            if len(emotion_window) >= frame_window:
                emotion_window.pop(0)

            try:
                # 获得出现次数最多的分类
                emotion_mode = mode(emotion_window)
            except:
                continue

            # Draw the main text in orange
            cv2.putText(frame, emotion_mode, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)

            # Draw slightly offset text to create a bold effect in orange
            cv2.putText(frame, emotion_mode, (x + 1, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)
            cv2.putText(frame, emotion_mode, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)

        print(frame.shape)

        cv2.imshow('FRAME', frame)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()
```
