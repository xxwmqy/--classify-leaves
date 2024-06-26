## 深度学习课程设计

kaggle 竞赛-classify_leaves
https://www.kaggle.com/competitions/classify-leaves

### 题目

内容：该任务是预测叶子图像的类别。该数据集包含 176 个类别，18353 张训练图像，8800 张测试图像。每个类别至少有 50 张用于训练的图像。

文件描述：
train.csv - 训练集
test.csv - 测试集
sample_submission.csv - 正确格式的示例提交文件（标签是随机生成的，因此初始精度约为 1/176）
images/ - 文件夹包含所有图像，

数据字段
image - 图像路径，例如 images/0.jpg
label - 类别名称

### 总体设计思路

#### 1. 数据预处理

在深度学习模型中，数据预处理是至关重要的一步。数据预处理的目标是将原始图像数据转换为适合模型输入的格式，并增强数据的多样性，以提高模型的泛化能力。我们对图像数据进行了以下几步预处理：

- **调整图像大小**：将所有图像调整为相同的尺寸，例如 224x224，以适应 ResNet-18 模型的输入尺寸。
- **数据增强**：在训练过程中，进行随机水平翻转和随机垂直翻转，以增加数据的多样性。
- **归一化**：将图像像素值归一化到[0, 1]范围，提高训练稳定性和模型性能。

#### 2. 数据集类定义

定义了一个自定义数据集类`LeavesDataset`，用于从 CSV 文件中读取图像文件名和标签，并加载图像数据。数据集类的设计考虑了训练、验证和测试三种模式，分别用于模型的训练、验证和测试。

#### 3. 数据加载器

使用 PyTorch 的`DataLoader`类，将数据集分批加载到模型中。数据加载器支持数据的随机打乱、批量加载和多线程加速读取。

#### 4. 模型选择与初始化

选择了预训练的 ResNet-18 模型，并进行了迁移学习。具体步骤如下：

- **加载预训练模型**：使用 PyTorch 的`torchvision.models`模块加载 ResNet-18 模型。
- **修改最后一层全连接层**：将模型的最后一层全连接层修改为输出类别数相匹配的层。
- **特征提取**：根据需要冻结卷积层参数，仅训练最后的全连接层。

#### 5. 模型训练

模型训练包括以下步骤：

- **定义损失函数和优化器**：使用交叉熵损失函数和 Adam 优化器。
- **训练过程**：在每个 epoch 中，进行前向传播、计算损失、反向传播和参数更新。同时在验证集上评估模型性能，监控损失值和准确率。
- **保存模型**：将训练好的模型参数保存到文件中，以便后续使用。

#### 6. 模型验证与测试

在验证集上评估模型性能，通过准确率和混淆矩阵分析模型的表现。最终在测试集上进行预测，生成分类结果。
