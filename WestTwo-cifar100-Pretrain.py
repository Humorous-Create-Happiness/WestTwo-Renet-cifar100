import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import resnet34
from torchvision.transforms.transforms import Resize

trainAcc=[]
trainLoss=[]

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])
# 采用自带的Cifar100
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)
net = models.resnet34(weights=None)
##迁移学习
for param in net.parameters():  # 固定参数
    print(param.names)
    param.requires_grad = False



fc_inputs = net.fc.in_features  # 获得fc特征层的输入
net.fc = nn.Sequential(  # 重新定义特征层，根据需要可以添加自己想要的Linear层
    nn.Linear(fc_inputs, 100),  # 多加几层都没关系

    nn.LogSoftmax(dim=1)
)

net.load_state_dict(torch.load('resnet34cifar100.pkl'))  # 装载上传训练的参数  #这行是新的

net = net.to('cuda')
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


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

        print(batch_idx + 1, '/', len(trainloader), 'epoch: %d' % epoch, '| Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    trainAcc.append(100. * correct / total)
    trainLoss.append(train_loss / (batch_idx + 1))
    print(trainAcc)
    print(trainLoss)

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
            print(batch_idx, '/', len(testloader), 'epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


for epoch in range(10):  # 设置成100循环
    train(epoch)
torch.save(net.state_dict(), 'resnet34cifar100.pkl')  # 训练完成后保存模型，供下次继续训练使用---86
print('begin  test ')

#for epoch in range(5):  # 测试5次
    #test(epoch)

#1:3.012     3.054
#2:3.796     4.675
#3:4.318     5.972
#4:4.123     6.73
#5:4.227     7.39
#6:996ciyacg.com
#7:
#8:
#9:
#0:
#11:
#12:
#13:
#14:
#15:
#16:
#17:
#18:
#19:
#20:
#21:
#22:
#23:
#24:
#25:
#26:
#27:
#28:
#29:
#30:
#31:
#32:
#33:
#34:
#35:
#36:
#37:
#38:
#39:
#40:
#41:
#42:
#43:
#44:
#45:
#46:
#47:
#48:
#49:
#50:
#51:
#52:
#53:
#54:
#55:
#56:
#57:
#58:
#59:
#60:
#61:
#62:
#63:
#64:
#65:
#66:
#67:
#68:
#69:
#70:
#71:
#72:
#73:
#74:
#75:
#76:
#77:
#78:
#79:
#80:
#81:
#82:
#83:
#84:
#85:
#86:
#87:
#88:
#89:
#90:
#91:
#92:
#93:
#94:
#95:
#96:
#97:
#98:
#99:
#100:
