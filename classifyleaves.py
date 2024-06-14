import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

# 数据预处理，数据集类
train_path = 'train.csv'
train_data = pd.read_csv(train_path, header=0)
print(train_data.head())
train_data.describe()
num_unique = train_data['label'].unique()
num_label = len(num_unique)  # 分为176类

train_label = sorted(list(set(train_data['label'])))
num_label = len(train_label)
# 名称转化为数值标签
label_to_num = dict(zip(train_label, range(num_label)))
# 数值标签转化为名称
num_to_label = dict(zip(range(num_label), train_label))
# 测试一下
print(num_to_label[0])
print(label_to_num[num_to_label[0]])


class LeavesDataset():
    def __init__(self, csv_path, img_path, mode='train'):
        """
        Args:
            csv_path (string): csv 文件路径

            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
        """
        self.mode = mode
        data_info = pd.read_csv(csv_path)
        data_len = len(data_info)
        self.img_path = img_path

        if mode == 'test':
            image_arr = np.array(data_info.iloc[:, 0])
            self.image_arr = image_arr
        else:
            # 验证集：9 / 10；训练集：1 / 10
            if mode == 'train':
                indices = [i for i in range(data_len) if i % 10 != 0]
            elif mode == 'valid':
                indices = [i for i in range(data_len) if i % 10 == 0]
            image_arr = np.array(data_info.iloc[indices, 0])
            label_arr = np.array(data_info.iloc[indices, 1])
            self.image_arr = image_arr
            self.label_arr = label_arr
        self.real_len = len(self.image_arr)
        print('Finished reading the {} set of the LeavesDataset ({} samples found)'.format(mode, self.real_len))
        
    # 重写__getitem__函数
    def __getitem__(self, index):
        assert index <= len(self)
        single_image_name = self.image_arr[index]
        image = Image.open(self.img_path + single_image_name)
        if self.mode == 'train':
            transform = transforms.Compose(
                [transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ToTensor()
                ])
        else:
            # valid test 不做数据增强
            transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        image = transform(image)

        if self.mode == 'test':
            return image
        else:
            label = self.label_arr[index]
            number_label = label_to_num[label]
            return image, number_label
        
    def __len__(self):
        return self.real_len


def prep_dataloader(train_path, img_path, mode, batch_size):
    dataset = LeavesDataset(train_path, img_path, mode)
    dataloader = DataLoader(dataset, batch_size, shuffle=(mode == 'train'), drop_last=False)
    return dataloader

# ---模型初始化和训练---
# 不需要训练的层
def set_parameter_requires_grad(model, feature_extracting):
    '''
    冻结模型的参数层，将对应的参数梯度设为零，不能反向传播
    '''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad =False


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_model(num_classes, feature_extract=False, pretrained=True):  
    # 加载预训练的ResNet18模型  
    model_ft = models.resnet18(pretrained=pretrained)  
  
    # 修改模型的全连接层以适应数据集  
    num_ftrs = model_ft.fc.in_features  
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)  
  
    if feature_extract:  
        # 如果进行特征提取，则冻结所有卷积层  
        for param in model_ft.parameters():  
            param.requires_grad = False  
        # 但确保全连接层是可训练的  
        model_ft.fc.requires_grad = True  
  
    return model_ft


# 绘画学习曲线
def plot_learning_curve(loss_record):
    total_steps = len(loss_record['train'])
    x = range(1,total_steps + 1)
    plt.figure(figsize=(6,4))
    plt.plot(x, loss_record['train'], 'r' ,label='train')
    plt.plot(x, loss_record['valid'], 'g', label='valid')
    plt.xlabel('Training epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend()
    plt.show()


def train(train_path, img_path, model, batch_size, epoch):
    # 数据集
    train_dataset = prep_dataloader(train_path, img_path, mode='train', batch_size=batch_size)
    valid_dataset = prep_dataloader(train_path, img_path, mode='valid', batch_size=batch_size)
    device = get_device()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    loss_record = {'train': [], 'valid': []}
    # 优化器
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = criterion.to(device)
    model = model.to(device)

    # 训练开始
    for i in range(epoch):
        print("--------------第 {} 轮训练开始-----------------".format(i+1))
        train_step = 0
        valid_step = 0
        a_epoch_train_loss = 0
        a_epoch_valid_loss = 0
        # 训练
        model.train()
        for img, label in train_dataset:
            img, label = img.to(device), label.to(device)
            train_label = model(img)
            train_loss = criterion(train_label, label)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_step += 1
            a_epoch_train_loss += train_loss
            a_epoch_train_loss_averge = a_epoch_train_loss / train_step
        loss_record['train'].append(a_epoch_train_loss_averge.detach().cpu().item())

    # 验证评估并保存模型
        model.eval()
        with torch.no_grad():
            for img, label in valid_dataset:
                img, label = img.to(device), label.to(device)
                train_label = model(img)
                valid_loss = criterion(train_label, label)
                valid_step += 1
                a_epoch_valid_loss += valid_loss
                a_epoch_train_loss_averge = a_epoch_valid_loss / valid_step
            loss_record['valid'].append(a_epoch_train_loss_averge.detach().cpu().item())
    torch.save(model.state_dict(), "model.pth")
    return loss_record


train_path = 'train.csv'
img_path = './'
model = initialize_model(num_label, feature_extract=False, pretrained=True)
batch_size = 32
epoch = 5
loss_record = train(train_path, img_path, model, batch_size, epoch)
# 画学习曲线
plot_learning_curve(loss_record)


# ---模型验证与测试---
model = model.to(get_device())
model.load_state_dict(torch.load('model.pth'))
model.eval()
# 迭代器
# 定义 DataLoader
valid_loader = prep_dataloader(train_path, img_path, mode='valid', batch_size=batch_size)

# 遍历 DataLoader
total = 0  
correct = 0  
for batch_idx, (img, label) in enumerate(valid_loader):  
    img, label = img.to(get_device()), label.to(get_device())  
  
    # 模型推理  
    with torch.no_grad():  
        output = model(img)  
        predicted_label = output.argmax(dim=1)  
  
    # 比较预测结果和标签  
    correct += (predicted_label == label.to(predicted_label.device)).sum().item()  
    total += label.size(0)  
  
    # 可以在每个批次后打印批次的精度（可选）  
    batch_accuracy = 100 * correct / total  
    print(f'Batch {batch_idx}: Accuracy: {batch_accuracy:.2f}%')  
  
# 计算整个验证集的精度（在循环结束后）  
validation_accuracy = 100 * correct / total  
print(f'Validation Accuracy: {validation_accuracy:.2f}%')

# 预测
model.eval()
pred = []
test_path = 'test.csv'
test_dataset = prep_dataloader(test_path, img_path, mode='test', batch_size=batch_size)
for img in test_dataset:
    img = img.to(get_device())
    with torch.no_grad():
        test_label = model(img)
    pred.extend(test_label.argmax(-1).cpu().numpy().tolist())
pred_to_str = [num_to_label[i] for i in pred]
test_data = pd.read_csv(test_path)
test_data['label'] = pred_to_str
test_data.to_csv('sample_submission.csv', index=False)
