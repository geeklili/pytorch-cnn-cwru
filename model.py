import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(stride=2, kernel_size=2)
            )
        self.fc = torch.nn.Sequential(torch.nn.Linear(1024, 32, dtype=torch.float64),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(32, 10, dtype=torch.float64))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        y_hat = torch.nn.functional.softmax(x, dim=1)
        return y_hat
