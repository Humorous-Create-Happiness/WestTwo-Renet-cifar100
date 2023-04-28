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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 默认的标准化参数
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)  # 导入训练集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True) #batchsize修改了

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)  # 导入测试集
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

net = models.resnet34(weights=None)  # pretrained=False or True 不重要,在0.15版本后改为weights=None
fc_inputs = net.fc.in_features  # 保持与前面第一步中的代码一致
net.fc = nn.Sequential(  #
    nn.Linear(fc_inputs, 100),  #
    nn.LogSoftmax(dim=1)
)

net.load_state_dict(torch.load('resnet34cifar100-ces.pkl'))  # 装载上传训练的参数
mydict = net.state_dict()
# for k,v in mydict.items():
#    print('k===',k,'||||,v==',v)

models = net.modules()
for p in models:
    if p._get_name() != 'Linear':
        print(p._get_name())
        p.requires_grad_ = False

net = net.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0008, momentum=0.9)  # 减小 lr


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




'''for epoch in range(4):

    if hasattr(torch.cuda, 'empty_cache'):       
        torch.cuda.empty_cache()#---新的

    train(epoch)
torch.save(net.state_dict(), 'resnet34cifar100.pkl')  # 训练完成后保存模型，供下次继续训练使用：35

print('begin  test ')'''
for epoch in range(1):
    test(epoch)


#cifar-100-python.tar.gz