import torch 
from torchsummary import summary 

class TA(torch.nn.Module):
    def __init__(self):
        super(TA, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.leakyrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2) # padding 'same' do not exit...
        
        self.dropout = torch.nn.Dropout(0.5)

        self.dense = torch.nn.Linear(1*1*32, 10)
        
    def forward(self, x):
        #one 
        output = self.conv1(x)
        output = self.leakyrelu(output)
        output = self.max_pool(output)
        #two
        output = self.conv2(output)
        output = self.leakyrelu(output)
        output = self.max_pool(output)
        #three
        output = self.conv3(output)
        output = self.leakyrelu(output)
        output = self.max_pool(output)
        #four
        output = self.conv4(output)
        
        output = self.dense(output.view(output.size(0), -1))
        
        return output
myMod = TA()
print(myMod)