import torch
import torch.nn as nn
# from torch_geometric.nn import GCNConv
# from utils import MiniTools
# from torch_geometric.datasets import Planetoid
#
#
#
#
# class GCN(nn.Module):
#     def __init__(self, input_size,graph_form):
#         super().__init__()
#         self.graph_form = graph_form
#         self.node_number = len(graph_form.x)
#         self.edge_number = len(graph_form.edge_attr)
#         self.edge_index = self.graph_form.edge_index #.cuda()
#         self.linear1 = nn.Linear(input_size, self.node_number)
#         #self.linear1 = nn.Linear(input_size, self.node_number + self.edge_number) # 线性变换
#         self.gconv1 = GCNConv(1, 1)
#         self.gconv2 = GCNConv(1, 1)
#         self.gconv3 = GCNConv(1, 1)
#         self.relu = nn.ReLU(True)
#         self.sigmoid = nn.Sigmoid()
#
#
#     def forward(self, x):
#
#         # 1). generate x_feature
#         node_feature = x.x
#         # 2). generate edge_index
#         #edge_select = temp_x[-self.edge_number:]>0.5
#         #edge_index = self.edge_index[:,edge_select]
#         #3. generate edge_attr -> experiment2中使用
#         #edge_attr = self.sigmoid(edge_select)
#         #4. GCN->relu->GCN->relu->GCN-> (W-GAN)
#         node_feature = self.gconv1([[0.1],[0.1]], x.edge_index)
#         node_feature = self.sigmoid(node_feature)
#         # node_feature = self.gconv2(node_feature, self.edge_index)
#         # node_feature = self.relu(node_feature)
#         # node_feature = self.gconv3(node_feature, self.edge_index)
#         #print(node_feature.detach().cpu().numpy().T)
#         #5. resize the x
#         out = torch.stack(batch_x,1).T
#         out = out.view(x.size(0),1,-1)
#         return out
#
