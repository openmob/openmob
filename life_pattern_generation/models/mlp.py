from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn

class WGANGP_Generator(nn.Module):

    def __init__(self,INPUTD_DIM,DIM,LP_DIM):
        super(WGANGP_Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(INPUTD_DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, LP_DIM),
            #nn.Sigmoid()
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output

class MLP_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize * isize),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize * isize, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)


class MLP_G_fcn2graph(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G_fcn2graph, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize),
            #nn.Sigmoid()
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, 1, self.isize)

class MLP_D_fcn2graph(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D_fcn2graph, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize, ndf),
            #nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            #nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            #nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)

class MLP_G_fcn2graph_noW(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super(MLP_G_fcn2graph_noW, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ngf
            nn.Linear(nz, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, ngf),
            nn.ReLU(True),
            nn.Linear(ngf, nc * isize),
            #nn.Sigmoid()
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, 1, self.isize)


class MLP_D_fcn2graph_noW(nn.Module):
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super(MLP_D_fcn2graph_noW, self).__init__()
        self.ngpu = ngpu

        main = nn.Sequential(
            # Z goes into a linear of size: ndf
            nn.Linear(nc * isize, ndf),
            #nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            #nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, ndf),
            #nn.BatchNorm1d(ndf),
            nn.ReLU(True),
            nn.Linear(ndf, 1),
            nn.Sigmoid()
        )
        self.main = main
        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = output.mean(0)
        return output.view(1)



#ConditionGAN for MLP
class MLP_G_fcn2graph_C(nn.Module):
    def __init__(self, isize0,isize, n_condtion, nz, nc, ngf, ngpu):
        super(MLP_G_fcn2graph_C, self).__init__()
        self.ngpu = ngpu


        self.fc1_1 = nn.Linear(nz, ngf)
        self.fc1_1_bn = nn.BatchNorm1d(ngf)
        self.fc1_2 = nn.Linear(n_condtion, ngf)
        self.fc1_2_bn = nn.BatchNorm1d(ngf)
        self.fc2 = nn.Linear(ngf*2, ngf*2)
        self.fc2_bn = nn.BatchNorm1d(ngf*2)
        self.fc3 = nn.Linear(ngf*2, ngf*4)
        self.fc3_bn = nn.BatchNorm1d(ngf*4)
        self.fc4 = nn.Linear(ngf*4, nc * isize0 * isize)
        self.Relu = nn.ReLU()

        self.nc = nc
        self.isize0 = isize0
        self.isize = isize
        self.nz = nz

    def forward(self, input,condition):
        input = input.view(input.size(0), input.size(1))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            print('Could not employ multi-processing in this mode.')
        else:
            # x = self.Relu(self.fc1_1_bn(self.fc1_1(input)))
            # y = self.Relu(self.fc1_2_bn(self.fc1_2(condition)))
            # x = torch.cat([x, y], 1)
            # x = self.Relu(self.fc2_bn(self.fc2(x)))
            # x = self.Relu(self.fc3_bn(self.fc3(x)))
            # x = self.Relu(self.fc4(x))

            x = self.Relu(self.fc1_1(input))
            y = self.Relu(self.fc1_2(condition))
            x = torch.cat([x, y], 1)
            x = self.Relu(self.fc2(x))
            x = self.Relu(self.fc3(x))
            x = self.Relu(self.fc4(x))

        return x.view(x.size(0), self.nc, self.isize0, self.isize)

class MLP_D_fcn2graph_C(nn.Module):
    def __init__(self, imageSize_0,isize, n_condition,nz, nc, ndf, ngpu):
        super(MLP_D_fcn2graph_C, self).__init__()
        self.ngpu = ngpu

        self.fc1_1 = nn.Linear(imageSize_0*isize, ndf*4)
        self.fc1_2 = nn.Linear(n_condition, ndf*4)
        self.fc2 = nn.Linear(ndf*8, ndf*2)
        self.fc2_bn = nn.BatchNorm1d(ndf*2)
        self.fc3 = nn.Linear(ndf*2, ndf)
        self.fc3_bn = nn.BatchNorm1d(ndf)
        self.fc4 = nn.Linear(ndf, 1)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Sigmoid = nn.Sigmoid()

        self.nc = nc
        self.isize = isize
        self.nz = nz

    def forward(self, input,condition):
        input = input.view(input.size(0),input.size(1) * input.size(2) * input.size(3))
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            print('Could not employ multi-processing in this mode.')
        else:
            x = self.LeakyReLU(self.fc1_1(input))
            y = self.LeakyReLU(self.fc1_2(condition))
            x = torch.cat([x, y], 1)
            # x = self.LeakyReLU(self.fc2_bn(self.fc2(x)))
            # x = self.LeakyReLU(self.fc3_bn(self.fc3(x)))
            x = self.LeakyReLU(self.fc2(x))
            x = self.LeakyReLU(self.fc3(x))
            x = self.Sigmoid(self.fc4(x))

        output = x.mean(0)
        return output.view(1)