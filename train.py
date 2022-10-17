import time
import model
import torch
import torch.nn as nn
from dataset import MyDataset
from torch.utils.data import DataLoader

# 超参数
batch_size = 128
num_epochs = 1000
learning_rate = 5e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 获取数据集
train_dataset = MyDataset()

# 定义模型
train_model = model.Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)

train_size = int(len(train_dataset) * 0.8)
valid_size = int(len(train_dataset)) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def test():
    train_model.load_state_dict(torch.load('./data/best.pt'))
    # torch.save(net, 'data/best_model.pt')
    train_model.eval()
    valid_acc = get_valid_acc(train_model)
    print(valid_acc)


def get_valid_acc(valid_model):
    valid_model.eval()
    li = list()
    for x_valid, y_valid in valid_loader:
        x_valid = x_valid.to(device)
        y_valid = y_valid.to(device)
        y_hat = valid_model(torch.tensor(x_valid, dtype=torch.float64))
        y_hat = torch.argmax(y_hat, 1)
        y = torch.argmax(y_valid, 1)
        res = y_hat == y
        res2 = res.int()
        acc = res2.sum().item() / batch_size
        li.append(acc)
    valid_model.train()
    return sum(li) / len(li)


def train():
    total_step = len(train_loader)
    acc_valid_old = 0
    a = time.time()
    for epoch in range(num_epochs):
        li = list()
        for i, (trains, labels) in enumerate(train_loader):
            trains = trains.to(device)
            labels = labels.to(device)
            # 预测和损失
            outputs = train_model(trains)
            loss = criterion(outputs, labels)
            # 计算准确率
            y_hat = torch.argmax(outputs, 1)
            y = torch.argmax(labels, 1)
            res = y_hat == y
            res2 = res.int()
            acc = res2.sum().item() / batch_size
            li.append(acc)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                acc_valid = get_valid_acc(train_model)
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.4f}  acc: {} valid_acc: {}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), sum(li) / len(li), acc_valid))

                if acc_valid > acc_valid_old:
                    torch.save(train_model.state_dict(), 'data/best.pt')
                    acc_valid_old = acc_valid
            li = list()
        b = time.time()
        print(b - a)
        a = time.time()


if __name__ == '__main__':
    # train()
    test()
