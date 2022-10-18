import torch.nn as nn


# 定义判别器  #####Discriminator######使用多层网络来作为判别器
# 将20371，通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。

class Discriminator(nn.Module):
    # input = 输入矩阵reshape为-1的size
    def __init__(self, input_size):
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


# ###### 定义生成器 Generator #####
# 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# 然后经过线性变换将其变成20371维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# 能够在-1～1之间。
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(input_size, input_size * 2),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(input_size * 2, input_size * 2),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(input_size * 2, output_size),  # 线性变换
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
        )

    def forward(self, x):
        x = self.gen(x)
        return x
