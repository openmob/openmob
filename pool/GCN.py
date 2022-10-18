import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将20371，通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class Discriminator(nn.Module):
    # input = 输入矩阵reshape为-1的size
    def __init__(self,input_size):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_size, 256),  # 输入特征数为1 x 20371 输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = self.dis(x)
        return x


class graphDiscriminator(nn.Module):
    def __init__(self, in_channels,graph):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 1)
        self.conv2 = GCNConv(2, 4)
        self.conv3 = GCNConv(4, 2)
        self.conv4 = GCNConv(2, 1)
        self.linear1 = nn.Linear(len(graph.x), 1)
        self.sigmoid = nn.Sigmoid()
        self.graph = graph

    def forward(self, x):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(self.graph.x, self.graph.edge_index).relu()
        #x = self.conv2(x, self.graph.edge_index).relu()
        #x = self.conv3(x, self.graph.edge_index).relu()
        #x = self.conv4(x, self.graph.edge_index).relu()
        x = self.linear1(x.view(-1))
        x = self.sigmoid(x)  # 也是一个激活函数，二分类问题中，
        return x

# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成20371维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self,input_size,output_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, output_size),  # 线性变换
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.gen(x)
        return x

class graphGenerator(nn.Module):
    def __init__(self, input_size,graph):
        super().__init__()
        self.graph = graph
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 512)  # 线性变换
        self.linear3 = nn.Linear(512, len(graph.x)*4)  # 线性变换
        self.gconv1 = GCNConv(4, 2)
        self.gconv2 = GCNConv(2, 1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #1. generate features from fnn
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        #2. restruct the feature x
        #x = x.view(4,len(self.graph.x))
        x = x.view(len(self.graph.x),4)
        #3. calcu y from x via x
        x = self.gconv1(x, self.graph.edge_index).relu()
        x = self.gconv2(x, self.graph.edge_index)
        x = self.sigmoid(x)
        return x.view(-1)

class graphGenerator_1(nn.Module):
    def __init__(self, input_size,graph):
        super().__init__()
        self.graph = graph
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 512)  # 线性变换
        self.linear3 = nn.Linear(512, len(graph.x)*2 + len(self.graph.edge_index[0]))  # 线性变换
        self.gconv1 = GCNConv(4, 2)
        self.gconv2 = GCNConv(2, 1)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        #1. generate features from fnn
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        #2. generate edge
        edge_select = self.sigmoid(x.view(-1)[0:len(self.graph.edge_index[0])])
        edge_index = self.graph.edge_index[:,edge_select>0.5]
        #3. restruct the feature x
        x = x.view(-1)[len(self.graph.edge_index[0]):].view(len(self.graph.x),2)
        #4. calcu y from x via x
        x = self.gconv2(x, edge_index)
        x = self.sigmoid(x)
        return x.view(-1)