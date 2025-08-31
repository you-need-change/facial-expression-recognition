# 人脸表情识别
## 1、背景
目前人们对表情识别的研究大都是从静态图像或者图像序列中提取表情特征，然后根据这些特征信息通过设计相应的分类算法把表情归为某一类别以完成分类任务。大多是分为三步，第一步预处理及人脸获取，第二步，特征提取，第三步，归类。将重点放在人脸整体上，将人脸整体进行分析处理，算法核心是降维处理。本文在此基础上将特征提取分为两步，将重点放在表情关键部分。人脸表情主要体现在眼睛、鼻子、嘴巴、眉毛四个关键部分，通过对这四个关键部分的定位，可以将人脸特征选择缩小至这些部位中。减少了大量计算量，同时也提高了识别率。先人脸特征点定位出人脸区域的关键部位，再对人脸区域的关键部位进行特征提取，而非对整体进行特征提取。经过特征提取，大量的冗余的数据被去除，空间维数大大降低。  
  
本次实验分为以下四部分：  

人脸检测识别   找出图像中人脸的大概区域  

人脸特征点定位   更加准确的定位出眼睛/眉毛等一些人脸区域的关键部位  

表情特征提取   面部表情识别的核心步骤  
  
表情分类   通过设置机制对表情进行分类，将表情归入相应类别  
  
面部表情识别的基本框架如下：  
<img width="802" height="685" alt="2025-08-31-224953" src="https://github.com/user-attachments/assets/7c89c904-6662-4650-83b0-1747996483e5" />
   
## 2、人脸表情识别模型设计
### 2.1 数据集
FER + 数据集是原始面部表情识别（FER）数据集的一个重要扩展,数据集来自Kaggle平台，下面是数据集的连接：  https://www.kaggle.com/datasets/msambare/fer2013  
为了改进原始数据集的局限性，FER + 提供了更精细和细致的面部表情标签。虽然原始FER数据集将面部表情分类为六种基本情绪——快乐、悲伤、愤怒、惊讶、恐惧和厌恶——但FER + 根据这一基础更进一步引入了两个额外的类别：中性和蔑视。  
<img width="698" height="139" alt="2025-08-31-202632" src="https://github.com/user-attachments/assets/f6007bbb-b546-485d-b8ff-17a01b25be4c" />

数据集的文件布局为分为7个类，格式如下：
<img width="1359" height="568" alt="屏幕截图 2025-08-31 230336" src="https://github.com/user-attachments/assets/36543c67-d9a3-403a-b1a7-ae172af2daf5" />


### 2.2 数据集加载
在人脸表情识别项目中，这段代码是数据预处理的关键一步，它负责将原始图像数据及其对应的表情类别信息读取到内存中，为后续的模型训练、验证和测试提供输入。通过这种方式，模型能够学习图像与表情标签之间的对应关系。这段代码的目的是从指定的文件夹中加载人脸表情数据集。它遍历每个以表情命名的子文件夹，将子文件夹名作为标签，然后逐一读取其中的灰度图像。
```
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
```
通过Dataloader类构建训练集、测试集
```
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```
```
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
### 2.3 模型设计
本项目使用的是基于CNN（卷积神经网络）的网络模型。构建的模型如下，输入图片 shape 为 48X48X1，经过一个3X3X32卷积核的卷积操作，经过BatchNorm操作和Relu激活层，经过5X5X32卷积操作和Relu激活层，经过2X2池化，以及dropout操作，得到一个 24X24X32 的 feature map 1（3X3卷积的padding为1，5X5卷积的padding为2，卷积的）。将 feature map 1经过一个 3X3X64 卷积核的卷积操作，以及BatchNorm操作和Relu激活层，经过5X5X64卷积操作和Relu激活层，再进行一次2X2的池化，以及dropout操作，得到一个 12X12X64 的 feature map 2。将feature map 2经过一个 3X3X128 卷积核的卷积操作，经过BatchNorm操作和Relu激活层，经过5X5X32卷积操作和Relu激活层，再进行一次 2X2 的池化，以及dropout操作，得到一个 6X6X128 的feature map 3。卷积完毕，数据即将进入全连接层。进入全连接层之前，要进行数据扁平化，将feature map 3拉一个成长度为 6X6X128=4608 的一维 tensor。
```
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
```
### 2.4 数据集的使用
在项目中，train() 函数是核心部分，它的作用是让模型从大量数据中学习特征，逐步提高其识别各种人脸表情的能力。训练一般都采用以下几步操作，前向传播、计算损失、反向传播与优化。然后在最后保存模型，方便以后进行调用和部署。最后做性能监测，打印当前训练周期的平均损失和准确率。
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
在项目中，test() 函数的作用是客观地衡量模型性能。它使用独立于训练过程的数据来评估模型的泛化能力，确保模型在未见过的新数据上也能表现良好，从而验证模型的有效性和可靠性。测试需要做以下操作，测试和训练不同，不需要计算梯度，经过前向传播，性能计算来算出模型在测试集上的准确率。
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

### 2.5 模型评估
使用matplotlib将训练、测试中的模型的精确度绘制成图像，便于分析训练的结果。
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
```
模型训练的结果图如下，可以看出模型在训练集和测试集上均能有不错的结果，相比训练集的91.82%的精确度，测试集的精确度为60.09%。
![ac](https://github.com/user-attachments/assets/1ee4653e-aca6-4414-bd99-ed75182c1121)

## 3、人脸表情识别的实现
FER_live_cam() 函数对视频帧进行实时的面部情绪识别。首先，它设置了一个字典 emotion_dict，将数字情绪类别索引映射到可读的情绪标签。视频源被初始化，尽管也可以使用网络摄像头输入。该函数还初始化了一个输出视频写入器，用于保存带有情绪注释的处理过的帧。主要的情绪预测模型以pth格式保存，在使用nn.Module的 load_state_dict 方法读取并与cascade人脸检测模型一起加载。在逐帧处理视频时，人脸检测模型通过边界框识别出人脸。
检测到的人脸在输入情绪识别模型之前经过预处理，包括调整大小和转换为灰度图像。通过从模型的输出分数中选择最大值确定识别出的情绪，并使用 class_names 将其映射到标签。然后，在检测到的人脸周围添加矩形框和情绪标签，将帧保存到输出视频文件中，并实时显示。用户可以通过按下 ‘q’ 键停止视频显示。一旦视频处理完成或中断，资源如视频捕获和写入器将被释放，并关闭任何打开的窗口。
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
```
<img width="600" height="338" alt="2025-08-31-222420" src="https://github.com/user-attachments/assets/afe57dcf-5511-4d0f-a32b-e8008a2fbab2" />

<img width="600" height="338" alt="2025-08-31-222449" src="https://github.com/user-attachments/assets/20332f1b-a834-4cb7-90d6-8a8a45c1f518" />

## 4、总结
本次实验采用一种新的人脸情绪识别流程，重点放在表情关键部分。大多数的人脸情绪识别算法将重点放在人脸整体上，将人脸整体进行分析处理，算法核心是CNN。本次实验采用人脸关键部位特征提取算法，经过 cascade 算法能够精确的定位人脸关键点部位，给后期处理减少了大量的数据量。本次实验的人脸特征向量选择采用了CNN神经网络，因为神经网络擅长处理二维图形信息，可以很好表达表情的变化。本项目设计了人脸表情识别视频处理器，方便识别人脸表情，使用深度学习的方式来检测面部表情。
