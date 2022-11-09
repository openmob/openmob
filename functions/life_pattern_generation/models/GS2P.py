import torch
import torch.nn.init as init
from torch import nn
from torch_geometric.nn import GCNConv


class GS2P(nn.Module):
    def __init__(self, lstm_input_size, lstm_embedding_size, lstm_hidden_size, lstm_num_layers, lstm_output_size,
                 graph_form,graph_output_size,conv_mode='GCN'):

        super(GS2P, self).__init__()
        #1. ======================================LSTM part=============================================
        self.num_layers = lstm_num_layers
        self.hidden_size = lstm_hidden_size

        self.input = nn.Linear(lstm_input_size, lstm_embedding_size)
        self.rnn = nn.LSTM(input_size=lstm_embedding_size,
                           hidden_size=lstm_hidden_size,
                           num_layers=lstm_num_layers,
                           batch_first=True,
                           bidirectional=True)

        self.output = nn.Sequential(
            nn.Linear(lstm_hidden_size*2, lstm_embedding_size),
            nn.ReLU(),
            nn.Linear(lstm_embedding_size, lstm_output_size)
        )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

        # 2. ======================================GCN part=============================================
        self.gconv1 = GCNConv(lstm_input_size, int(lstm_input_size/2))
        self.gconv2 = GCNConv(int(lstm_input_size/2), int(lstm_input_size/4))
        self.gconv3 = GCNConv(int(lstm_input_size/4), 1)
        self.gcn_out = nn.Linear(graph_form.x.shape[0], 24) # 线性变换

        self.conv_mode = conv_mode
        if self.conv_mode =='FCN':
            self.conv1 = nn.Linear(lstm_input_size, int(lstm_input_size/2))
            self.conv2 = nn.Linear(int(lstm_input_size/2), int(lstm_input_size/4))
            self.conv3 = nn.Linear(int(lstm_input_size/4), 1)

        # 3. ======================================Merge part=============================================
        self.merge_out = nn.Linear(48+graph_form.x.shape[0], 2)
        self.sigmoid = nn.Sigmoid()




    def forward(self,lstm_input_raw, input_nodes,input_edges):
        #1. ======================================LSTM part=============================================
        #(1). encoding via fcn
        input = self.input(lstm_input_raw)
        input = self.relu(input)
        #(2). lstm
        output_raw, hidden = self.rnn(input, self.hidden)
        #(3). decoding via fcn
        lstm_out = self.output(output_raw)
        # for hidden in self.hidden:
        #     hidden = hidden.detach()
        #2. ======================================GCN part=============================================
        #(1). encoding via fcn
        if self.conv_mode == 'FCN':
            node_feature = self.conv1(input_nodes)
            node_feature = self.relu(node_feature)
            node_feature = self.conv2(node_feature)
            node_feature = self.relu(node_feature)
            node_feature = self.conv3(node_feature)
            node_feature = self.relu(node_feature)
        else:
            node_feature = self.gconv1(input_nodes, input_edges)
            node_feature = self.relu(node_feature)
            node_feature = self.gconv2(node_feature, input_edges)
            node_feature = self.relu(node_feature)
            node_feature = self.gconv3(node_feature, input_edges)
            node_feature = self.relu(node_feature)

        #gcn_out = self.gcn_out(node_feature.view(-1))

        #3. ======================================Merge part=============================================
        merge_out = torch.cat([lstm_input_raw[:,:,-1].view(-1),lstm_out.view(-1),node_feature.view(-1)],0)
        #print(lstm_input_raw[:,:,-1].view(-1))
        merge_out = self.merge_out(merge_out)
        merge_out = self.sigmoid(merge_out)
        base_pob = torch.prod(lstm_input_raw[:,:,-1].view(-1))
        adjust_pob = base_pob*merge_out[0] + merge_out[1]
        return merge_out,adjust_pob
    # def init_hidden(self, batch_size):
    #     return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
    #             Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())


# LSTM = WordLSTM.LSTM_plain(input_size=node_feature_num, embedding_size=embedding_size_rnn_output,
#                    hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
#                    has_output=True, output_size=1 * node_feature_num).to(device)
# seq_feature = LSTM(torch.unsqueeze(torch.FloatTensor(feat).to(device),0))

