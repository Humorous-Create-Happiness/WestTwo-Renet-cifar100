## python-AI-第三轮

## 自由选择算法完成对cifar-100数据集的分类

背景：Cifar-100数据集由20个粗类和100个细类组成，每个粗类包含5个细类，每个细类有500张训练图片和100张测试图片。

### STEP1：深入了解Retnet（残差网络）算法

#### 0.导入：

​         假设使用标准优化算法（梯度下降法等）训练一个普通网络，如果没有残差，没有这些捷径或者跳跃连接，凭经验你会发现随着网络深度的加深，训练错误会先减少，然后增多。而理论上，随着网络深度的加深，应该训练得越来越好才对，网络深度越深模型效果越好。但实际上，如果没有残差网络，对于一个普通网络来说，存在着**梯度消失/梯度爆炸**的问题。深度越深意味着用优化算法越难训练，随着网络深度的加深，训练错误会越来越多。

​		但有了ResNets就不一样了，即使网络再深，训练的表现却不错，比如说训练误差减少，就算是训练深达100层的网络也不例外。对x的激活，或者这些中间的激活能够到达网络的更深层。这种方式有助于解决梯度消失和梯度爆炸问题，在训练更深网络的同时，又能保证良好的性能。

​         ResNet的发明者是何恺明、张翔宇、任少卿和孙剑，他们发现使用**残差块**能够训练更深的神经网络。所以构建一个ResNet网络就是通过将很多这样的残差块堆积在一起，形成一个很深神经网络。首先回忆一个普通网络（Plain network），这个术语来自ResNet论文。



​     	0.1什么是**梯度消失/梯度爆炸**：

​				深度神经网络训练的时候，采用的是反向传播方式，该方式使用链式求导，计算每层梯度的时候会涉及一些连乘操作，因此如果网络过深。

​				那么如果连乘的因子大部分小于1，最后乘积的结果可能趋于0，也就是梯度消失，后面的网络层				的参数不发生变化.
​				那么如果连乘的因子大部分大于1，最后乘积可能趋于无穷，这就是梯度爆炸



​		0.2什么是**残差块(Residual block)**：

​        ![image-20230407063902010](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230407063902010.png)

​          上图是一个两层神经网络。回顾之前的计算过程,他在残差网络中有一点变化：



![image-20230407064054626](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230407064054626.png)

​         

​	    我们直接将a^[l]向后，到神经网络的深层，在ReLU非线性激活函数前加上a^[l]，将激活值a^[l]的信息直接传达到神经网络的深层，不再沿着主路进行，因此a^[l+2]的计算公式为：

​           **a^[l+2]=g(z^[l+2]+a^[l])**

​		加上a^[l]后产生了一个残差块（residual block）。插入的时机是在线性激活之后，ReLU激活之前。除了捷径（shortcut），你还会听到另一个术语“跳跃连接”（skip connection），就是指a^[l]跳过一层或者好几层，从而将信息传递到神经网络的更深层。

![](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230406064209434.png)

​		上图右侧为残差网络算法下的训练错误降低曲线



​		0.3什么是**全连接层**



​		全连接层 Fully Connected Layer 一般位于整个卷积神经网络的最后，负责将卷积输出的二维特征图转化成一维的一个向量，由此实现了端到端的学习过程（即：输入一张图像或一段语音，输出一个向量或信息）。全连接层的每一个结点都与上一层的所有结点相连因而称之为全连接层。由于其全相连的特性，一般全连接层的参数也是最多的。

​     	卷积层的作用只是提取特征，但是很多物体可能都有同一类特征，比如猫、狗、鸟都有眼睛。如果只用局部特征的话不足与确定具体类别。这时就需要使用组合特征来判别了。全连接层的作用就是组合这些特征来最终确定是哪一个分类，所以**全连接就是组合特征和分类器功能**。

​       实现方式：

![image-20230407070125563](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230407070125563.png)

​		在这里如上图所示，一个网络在全连接层之前，生成了5@3×3的特征映射，我们需要只需要使用五个卷积核去和激活函数的输出进行卷积运算，在将五个输出的值相加即可得到一个全连接层的输出值。如果结果是N维的向量，则需要N×5个3×3的卷积核。再加上求和运算对应的权值，参数的数量是非常巨大的，由此一般只在**网络的之后或者池化以后**使用全连接层且不建议多次使用。
















#### 1.ResNet34残差网络算法的定义（结构图如下）：

​         定义：ResNet34由1个卷积层，16个[**残差块**](https://so.csdn.net/so/search?q=残差块&spm=1001.2101.3001.7020)（在conv2_x残差层中有3个块，下同）和1个全连接层组成，其中在**全连接层前**做的是**平均池化**，而不是最大值池化。

![image-20230405071205062](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230405071205062.png)

上图为ImageNet架构。构建块显示在括号中，以及构建块的堆叠数量。下图为残差网络结构图

![image-20230408065951818](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230408065951818.png)

#### 2.残差网络的使用与构建

​	2.1 构建单个残差块

​		一个残差单元的结构如下。输入为X ；**weight layer 代表卷积层**，这里是指 **convolution卷积层 + batch normalization批标准化层** ；relu 是激活函数 ； identity 是将输入 X 经过变换后与卷积层的输出结果相加，下面会详细说明。

​	![image-20230408070153202](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230408070153202.png)



​	残差块中的第一个卷积层 self.conv1，主要用于下采样特征提取。

如果步长 strides=1，由于padding='same'，该层的输入和输出特征图的size不变。属于结构图中左侧蓝色部分。

如果步长 strides=2，表示该层输出的特征图的 size 是输入的特征图 size 的一半。由于卷积核的size是 3*3 ，卷积核移动时会出现滑窗无法覆盖所有的像素格的现象。可能会出现，该层的输出特征图size不等于输入size的一半。通过padding='same'自动填充输入图像，让输出size等于一半的输入size。属于结构图中的左侧后三种颜色的部分

​	残差块中的第二个卷积层 self.conv2，主要用于进一步提取特征，不进行下采样。

规定其步长 stride=1，由于padding='same'，该层的输入和输出特征图的size不变。

完成卷积部分convblock之后，接下来看短接部分identityblock

identity 负责将输入 X的shape 变换到和卷积部分的输出的shape相同。

​	如果第一个卷积层 self.conv1 的步长 strides=1，那么输入特征图的 shape 和卷积层输出的特征图的 shape 相同，这时 identity 不需要变换输入特征图 X 的shape。

​	如果第一个卷积层 self.conv1 的步长 strides=2，那么输入特征图的 size 变成了原来的一半。这时，为了能将输入 X 和 卷积层输出结果相加，需要通过 identity 重塑输入 X 的shape。这里使用的是 1*1 卷积传递特征，1*1的卷积核遍历所有的像素格后不会改变特征图的size，设置步长strides=2，成功将特征图的size变成原来的一半。属于结构图中的左侧后三种颜色的部分。

这样，我们就完成了对单个残差块中所有层的初始化，接下来将层之间的前向传播过程写在 call() 函数中。这里需要注意的就是 layers.add([out, identity]) ，将卷积层的输出特征图的结果和输入的特征图相加。identity 只负责将输入特征图的 shape 变换成和卷积部分输出特征图的 shape 相同。

代码如下：

```python
# Basic Bolck 残差块
# x--> 卷积 --> bn --> relu --> 卷积 --> bn --> 输出 
# |---------------Identity(短接)----------------|
 
# 定义子类，一个残差块
class BasicBlock(layers.Layer):  # 继承父类的方法和属性
    
    #（1）子类初始化
    # filter_num 代表传入卷积核数量，将输入图像的通道数变成残差块的规定的通道数
    # stride 代表步长，默认为1，代表不对输入图片的size采样，如果不做padding，得到的图像的size就略小，做padding后输入和输出的size保持一致
    # strdie=2时，代表二分采样，输出的size只有输入size的一半
    
    def __init__(self, filter_num, stride=1):
        
        # 继承父类的初始化方法，
        # super()中的第一个参数是子类名称，第二个是子类的实例化对象
        super(BasicBlock, self).__init__()
        
        # 在父类初始化的基础上添加新的属性
        
        # 卷积层1，传入卷积核数量，卷积核size，步长
        # 如果stride=1，为避免输出小于输入，设置padding='same'，使输入等于输出
        # 如果stride=2，若输入为32*32，由于卷积核3*3的影响，使输出不等于16*16，这时通过padding=same在输入图像上自动补全，如果输出小于16会自动补成16
        self.conv1 = layers.Conv2D(filter_num, (3,3), strides=stride, padding='same')
        
        # 标准化层batchnormalizeation
        self.bn1 = layers.BatchNormalization()
        
        # relu激活函数层，没有其他参数，可以作为一个函数使用多次。而有参数设置的一些层，只能单独对应使用
        self.relu = layers.Activation('relu')
        
        # 卷积层2，如果上一个卷积层stride=2完成下采样，那么这里的卷积层就不进行下采样了，保持stride=1
        self.conv2 = layers.Conv2D(filter_num, (3,3), strides=1, padding='same')
        
        # 标准化层
        self.bn2 = layers.BatchNormalization()
        
        
        # identity层需进行维度变换，将原始输入图像和卷积后的图像相匹配
        # 进行1*1卷积匹配通道数，通过stride匹配图像的size
        self.downsample = Sequential()  # 设置容器
        
        # 在容器中添加1*1卷积和步长变换
        # stride保持和第一个卷积层一致，保证convblock和identityblock能直接相加
        # 如果第一个卷积层的stride=1时，那么输入和输出的shape保持一致
        self.downsample.add(layers.Conv2D(filter_num, (1,1), strides=stride))
        
        
    #（2）前向传播
    # 定义类方法，self为类实例化对象
    def call(self, inputs, training=None):
        
        # 卷积层1，调用初始化后的属性
        x = self.conv1(inputs)  # 输入原始图像
        x = self.bn1(x)
        x = self.relu(x)
        
        # 卷积层2
        x = self.conv2(x)
        out = self.bn2(x)
 
        # identity层，输入的是原始输入图像
        identity = self.downsample(inputs)
        
        # 将convblock和identityblock相加得到最终的残差块的输出结果
        output = layers.add([out, identity])
        
        # 最终结果经过一个非线性函数
        output = tf.nn.relu(output)
        
        # 返回残差块的输出结果
        return output
```

​	2.2 叠加多个残差块

上面我们已经成功完成了一个残差块，然而一个残差结构是由多个残差块叠加而成的。下面是放大了的结构图，可见 resnet18 中每一个残差结构是由 2 个残差单元组合而成

   ![image-20230410063332478](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230410063332478.png)

我们定义一个函数 build_resblock 用来组合残差结构。这里需要注意的是，blocks 代表一个残差结构需要堆叠几个残差单元，resnet18 和 32 中是2个。看结构图可知，在残差结构中只有第一个残差单元具备下采样改变特征图 size 的能力。因此第一个残差块的步长 stride，需要根据输入来确定。而除第一个以外的残差块都不会改变特征图的size，因此固定步长stride=1。每一个残差结构的卷积核个数都是相同的，要通过输入来确定。 

```python
    # 利用单个已定义的残差块，叠加多个残差块
    # filter_num，代表当前图像的特征图个数
    # blocks，需要代表堆叠几个残差块
    # stride，代表当前的步长，等于1
    def build_resblock(self, filter_num, blocks, strides=1):
        
        # 使用Sequential容器装填网络结构
        res_blocks = Sequential()
        
        # 在ResNet类中对BasicBlock类实例化，构成组合关系
        # ResNet类可调用BasicBlock类中的所有属性和方法
        
        # 添加网络层
        # 第一个残差块有下采样功能，stride可能等于2
        res_blocks.add(BasicBlock(filter_num, strides))
        
        # 每个残差结构中剩余的残差块不具备下采样功能，stride=1
        for _ in range(1, blocks):
            
            # 残差结构中剩余的残差块保持图像的shape不变
            res_blocks.add(BasicBlock(filter_num, stride=1))
        
        # 返回构建的残差结构
        return res_blocks
```



​	2.3构建残差网络

上面我们已经完成了残差块的构建，现在我们需要做的就是将这些残差结构按顺序堆叠在一起就能组建残差网络。

首先我们看初始化函数中的代码。self.stem 是用来处理原始输入图像的，假设原始输入的shape为 [224, 224, 3]，根据网络结构图设置预处理卷积层的各个参数。通过最大池化 layers.MaxPool2D 指定步长为2，将预处理卷积层的特征图的size减半 



接下去就可以根据每个残差结构的配置参数，第一个残差结构 self.layer1 由图可知，没有进行下采样，因此步长 stride=1，第一个残差结构中的卷积核个数统一是64个，每个残差结构由2个残差单元组成 layer_dims=[2,2,2,2]，初始化时都是调用的上面定义的残差结构函数 build_resblock。

第二个残差结构 self.layer2 由图可知，第一个残差块进行了下采样，因此，要指定步长 strides=2，特征图的 size 减半，特征图的个数统一都是128。同理其他两个残差结构。

​    

最后将残差层的输出结果经过全局平均池化后放入全连接层，得出分类结果。 layers.GlobalAveragePooling2D() 是在通道维度上对w和h维度求平均值。将特征图的shape从 [b, w, h, c] 变成 [b, 1, 1, c] 

```py
# 定义子类ResNet，继承父类keras.Model
class ResNet(keras.Model):
    
    #（1）初始化
    # layer_dims=[2,2,2,2]，resnet18包含4个残差结构res_blocks，每个残差结构中有2个残差块
    # num_classes 代表最终的输出的分类数
    def __init__(self, layer_dims, num_classes=1000):  
        
        # 调用父类的初始化方法
        super(ResNet, self).__init__(self)
        
        # 分配属性
        # 原始图像输入的预处理卷积和池化
        self.stem = Sequential([layers.Conv2D(64, (7,7), strides=(2,2), padding='same'),  # 3*3卷积提取特征
                                layers.BatchNormalization(),      # 标准化 
                                layers.Activation('relu'),        # 激活函数
                                layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')])  # 最大池化，输入图像的size减半
        
        # 创建4个残差结构，layer_dims=[2,2,2,2]
        self.layer1 = self.build_resblock(64, layer_dims[0])  # 第一个残差结构指定64个卷积核，包含2个残差块
        self.layer2 = self.build_resblock(128, layer_dims[1], strides=2)  # 第二个残差结构128个卷积核，包含2个残差块，步长为2，图像的size减半
        self.layer3 = self.build_resblock(256, layer_dims[2], strides=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], strides=2)
        
        # 全局平均池化，不管卷积层输出的长和宽是多少，在channel维度上将所有的长和宽加起来取均值
        # [b, w, h, c] ==> [b,c]
        self.avgpool = layers.GlobalAveragePooling2D()
 
        # 全连接层用于图像分类
        self.fc = layers.Dense(num_classes)
 
 
    #（2）定义前向传播的类方法
    def call(self, inputs, training=None):
        
        # 原始输入经过预处理卷积层
        x = self.stem(inputs)
        
        # 经过4个残差结构
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 输出层
        x = self.avgpool(x)  # 输出shape[b,c] --> [None, 512]
        x = self.fc(x)       # 输出[b,1000]
        
        # 返回分类结果
        return x

```



### STEP2：使用残差网络解题



#### 1.使用Retnet34初步解题

拟定迁移的层和需要调整参数的层，这里只将fc层重新学习，其余各层的权重weight固定不变。将学习速率设置大一些（初始 lr=0.01），然后保存模型，代码及解释如下:

```py
import torch
import torch.nn as nn
import torch.optim as optim
  #torch.optim是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性，使得未来  	能够集成更加复杂的方法。

import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import resnet34
from torchvision.transforms.transforms import Resize
 
transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
	#由于数据集的图片类型是 PIL Image，torch 无法直接使用，所以要先转为 tensor，通过 transforms 实	现。这里用 transforms 将图片转为 tensor 类型后，并用 tensorboard 进行查看。
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    
    #尝试的代码：
    #transforms.RandomHorizontalFlip(),  # 随机水平镜像
    #transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    #transforms.RandomCrop(32, padding=4),  # 随机裁剪
    
])

##采用自带的Cifar100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)
 
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
net=models.resnet34(pretrained=True)      #为了加速学习速度，batch_size应为2的幂次方，（我的显卡很差，只能用32作为大小）

##迁移学习
for param in net.parameters(): #固定参数
    print(param.names)
    param.requires_grad = False
 
fc_inputs = net.fc.in_features #获得fc特征层的输入
net.fc = nn.Sequential(         #重新定义特征层，根据需要可以添加自己想要的Linear层
    nn.Linear(fc_inputs, 100),  #多加几层都没关系
    nn.LogSoftmax(dim=1)
)
 
net = net.to('cuda')
criterion = nn.CrossEntropyLoss()
  #nn.CrossEntropyLoss是pytorch下的交叉熵损失，用于分类任务使用
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  #SGD就是optim中的一个算法（优化器）：随机梯度下降算法，这里为了使用torch.optim，需要构建一个			optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。其中params 			(iterable) – 待优化参数的iterable（w和b的迭代）或者是定义了参数组的dict，lr (float) – 学习率		momentum (float, 可选) – 动量因子（默认：0）
 
##Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):#enumerate是枚举函数
        inputs, targets = inputs.to('cuda'), targets.to('cuda')#数据导入
        optimizer.zero_grad()
		#上一行 是 PyTorch 中的一个函数，用于清零优化器中所有参数的梯度。在训练神经网络的过程中，通常在每个训练步之前调用这个函数。因为在反向传播计算梯度之前，需要将之前计算的梯度清零，以免对当前计算造成影响
        outputs = net(torch.squeeze(inputs, 1))#将输入缩小为1
        
        loss = criterion(outputs, targets)     #简单的巡视函数
        loss.backward()                        #根据loss来计算网络参数的梯度
        optimizer.step()                       #针对计算得到的参数梯度对网络参数进行更新的优化器
        train_loss += loss.item()              #一般不直接使用loss，显存会炸，这里用item取一个元素张量里面的具体元素值并返回该值
        _, predicted = outputs.max(1)          #torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示，不重要可以忽略），第二个值是value所在的index（也就是predicted）。
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()#将正确的输入correct
 
        print(batch_idx+1,'/', len(trainloader),'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():#没有梯度就清除一次，释放一些GPU空间
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx,'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
for epoch in range(100): #设置成100循环
    train(epoch)
torch.save(net.state_dict(),'resnet34cifar100.pkl') #训练完成后保存模型，供下次继续训练使用
print('begin  test ')
 
for epoch in range(5): #测试5次
    test(epoch)
```

在100次epoch后，我们得到了一个初步的模型，存在resnet34cifar100.pkl之中，数据此时并不尽如人意，尽管学习率已经下降到0.001，训练集仍只有约17.5%的准确率，测试集更是只有5%。

尝试加入一些反转：

```python
    transforms.RandomHorizontalFlip(),  # 随机水平镜像
    transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    
```

效果微乎其微。



#### 2.使用反向学习继续解题

将保存的模型加载进来，降低学习速率继续学习（降低到 lr=0.001） ，注意代码有变化，仍然要固定前面的Conv2d层的参数，放开最后的fc层的参数反向学习功能。核心代码看下图：

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20230411063249091.png" alt="image-20230411063249091" style="zoom:200%;" />



改完后的代码如下：

```py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import resnet34
from torchvision.transforms.transforms import Resize
 
transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #默认的标准化参数
])
 
 
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)#导入训练集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)
 
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)#导入测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
 
net=models.resnet34(pretrained=False) # pretrained=False or True 不重要
fc_inputs = net.fc.in_features # 保持与前面第一步中的代码一致
net.fc = nn.Sequential(         #
    nn.Linear(fc_inputs, 100),  #
    nn.LogSoftmax(dim=1)
)
 
net.load_state_dict(torch.load('resnet34cifar100.pkl')) #装载上传训练的参数
mydict=net.state_dict()
#for k,v in mydict.items():    
#    print('k===',k,'||||,v==',v)
 
models=net.modules()
for p in models:
    if p._get_name()!='Linear':
        print(p._get_name())
        p.requires_grad_=False
 
net = net.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #减小 lr
 
# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = net(torch.squeeze(inputs, 1))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
 
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
 
        print(batch_idx+1,'/', len(trainloader),'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(torch.squeeze(inputs, 1))
            loss = criterion(outputs, targets)
 
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
 
            print(batch_idx,'/',len(testloader),'epoch: %d'% epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
 
 
for epoch in range(10):
    train(epoch)
torch.save(net.state_dict(),'resnet34cifar100.pkl') #训练完成后保存模型，供下次继续训练使用
 
print('begin  test ')
for epoch in range(5):
    test(epoch)
```

此时结果有了突破性进展，仅仅在10次epoch之后训练集就已经突破99.7%，而测试集也有47%的准确率。



#### 3.过拟合的出现与解决

接下来在经过50次的epoch后，我发现测试集准确率并没有提升，考虑到是过拟合的问题，于是在transform中加入以下代码：

```py
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#新的
    transforms.RandomHorizontalFlip(),  # 随机水平镜像
    transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
    transforms.RandomCrop(224,128),  # 随机裁剪,长与宽至少有一个接近Resize学习效果更好（不然就卡在5%力）
   # transforms.

])
```

在学习了100次epoch后，在训练集上达到了87.5%的正确率，而在原来（未加入随机代码）的测试集上跑达到了69.54%的正确率：

![屏幕截图 2023-04-28 180228](C:\Users\Lenovo\Desktop\屏幕截图 2023-04-28 180228.png)





#### 4.可视化
可以通过history日志调出训练时训练集和测试集的正确率和损失值并用matplotlib绘制图像（我忘了）：

```py
from matplotlib import pyplot as plt

fig1, ax_acc = plt.subplots()
plt.plot(history.history['sparse_categorical_accuracy'], 'r', label='acc')
plt.plot(history.history['val_sparse_categorical_accuracy'], 'b', label='val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(loc='lower right')
plt.show()

fig2, ax_loss = plt.subplots()
plt.plot(history.history['loss'], 'r', label='loss')
plt.plot(history.history['val_loss'], 'b', label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(loc='upper right')
plt.show()

```

最后得出训练集acc如下（纵坐标为正确率*100，横坐标为训练次数）：

![屏幕截图 2023-04-28 204912](C:\Users\Lenovo\Desktop\屏幕截图 2023-04-28 204912.png)

训练集loss如下：（纵坐标为loss，横坐标为训练次数）

![屏幕截图 2023-04-28 205737](C:\Users\Lenovo\Desktop\屏幕截图 2023-04-28 205737.png)







Refererces:

​                         https://blog.csdn.net/dgvv4/article/details/122396424

​                         https://blog.csdn.net/lillllllll/article/details/120144173

​                         https://blog.csdn.net/ereqewe/article/details/126636464

​           

