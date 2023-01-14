import numpy
import time

import numpy as np
import torch
# import torch_geometric.datasets as datasets
# import torch_geometric.data as data
# import torch_geometric.transforms as transforms
import networkx as nx
# from torch_geometric.tools.convert import to_networkx
import matplotlib.pyplot as plt
from utils import MiniTools
import pandas as pd
from pandas import DataFrame
from pyvis.network import Network
from tqdm.notebook import tqdm
import time
#from tensorboard_logger import configure, log_value
# from models import GCN
from models import WordLSTM



class LifePattern():
    '''
    This is for test.
    '''
    def __call__(self):
        return 1

    def __init__(self, G, index2node, node2index):
        '''
        This is for initial class.
        '''

        # 1. Get the initial global information for this G
        self.G = G
        self.EdgesPob = nx.get_edge_attributes(G, 'EdgePob')
        self.NodesValue = nx.get_node_attributes(G, 'NodePob')
        self.NodesStr = nx.get_node_attributes(G, 'KeyPointStr')
        self.NodesInt = np.array(list(G.nodes))
        self.hour_list = np.array([int(x.split('.')[-1]) for x in list(self.NodesStr.values())])
        self.index2node = index2node
        self.node2index = node2index
        # 2. Initial the results
        self.TravelResults = []
        # 3. Initial start possibility of traversal
        ### 基于规则的方法，暂时不用
        # start_candis = self.NodesInt[self.hour_list == 0]
        # start_candis = [self.NodesStr[x] for x in start_candis]
        # # 按照规则排序 0>1>2... H>W>0
        # start_candis_sort = []
        # for x in start_candis:
        #     order = int(x[2])
        #     type = x[0]
        #     if type == 'H':
        #         start_candis_sort.append(order*10 + 0)
        #     elif type == 'W':
        #         start_candis_sort.append(order*10 + 1)
        #     else:
        #         start_candis_sort.append(order * 10 + 2)
        # start_candis = [start_candis[n]  for n in np.argsort(start_candis_sort)]
        # start_candis_pob = np.zeros(len(start_candis))
        # start_candis_pob[0] =1
        # self.TravStartNodes = [start_candis, start_candis_pob]
        ### 基于统计概率的方法
        self.start_stat_pob = pd.read_csv('start_node_pob_table.csv',index_col=0)
        #start_node = self.start_stat_pob.index.values[self.wordSample(self.start_stat_pob.values[:,0])]
        start_candis = [x +'.0' for x in self.start_stat_pob.index.values]
        self.TravStartNodes_Prior = [start_candis,self.start_stat_pob.values[:,0]]
        self.TravStartNodes = self.TravStartNodes_Prior.copy()
        #self.visGraph(self.G)

    def wordSample(self,preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)#重新加权调整
        preds[np.isnan(preds)] = 0  #这里有时候会出现nan，后面有空再看为什么会有nan，先置0，不出bug
#         preds = preds +0.00001
        #print(preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)#返回概率最大的字符下标

    def getInit(self):
        init_hour = 0
        init_hour_idx = np.atleast_1d(np.squeeze(np.argwhere(self.hour_list == init_hour)))
        #print(init_hour_idx)
        while len(init_hour_idx) == 0:
            init_hour_idx = np.squeeze(np.argwhere(self.hour_list == init_hour)) # 得到当前小时内的node数组位置
            init_hour = init_hour +1
        if len(init_hour_idx) == 0:
            print('ERROR: COULD NOT FIND INITITIAL NODES.')
        if init_hour > 0 :
            print('WARNING: INITIAL HOUR NOT 0 CLOCK.')
        self.init_hour_idx = init_hour_idx
        self.init_hour = init_hour
        return init_hour_idx,init_hour

    def findNext(self,cr_node_int,cr_hour,temperature):
        next_neigh_nodes_str = []
        next_neigh_nodes_int = []
        if len(list(self.G.neighbors(cr_node_int)))>0:
            for next_state_int in list(self.G.neighbors(cr_node_int)):
                next_state_idx = np.argwhere(self.NodesInt==next_state_int)[0][0] #这里idx是相对位置，int是绝对int_label
                temp_hour = self.hour_list[next_state_idx]
                if temp_hour == (cr_hour + 1):
                    next_neigh_nodes_str.append(self.NodesStr[next_state_idx])
                    next_neigh_nodes_int.append(self.NodesInt[next_state_idx])
        else:
            #print('No more neighber nodes.')
            return None,None
        pob = [self.EdgesPob[(cr_node_int,x)] for x in next_neigh_nodes_int]
        pob = MiniTools.normSum(pob)
        select_node = self.wordSample(preds = pob, temperature=temperature)
        next_node_int = next_neigh_nodes_int[select_node]
        next_node_str = next_neigh_nodes_str[select_node]
        return next_node_int,next_node_str

    def travSeq(self,temperature=[1 for x in range(24)]):
        trav_seq = []
        init_hour_idx,init_hour = self.getInit()
        self.init_str2int_dict = {}
        for x in init_hour_idx:
            self.init_str2int_dict.update({self.NodesStr[x]: x})
        cr_node_int = self.wordSample(preds=self.TravStartNodes[1])
        cr_node_int = self.TravStartNodes[0][cr_node_int]
        while_count = 0
        print('loops')
        while cr_node_int not in list(self.init_str2int_dict.keys()):
            print('if qian')
            if while_count < 100:
                print('loops%d' % while_count)
                cr_node_int = self.wordSample(preds=self.TravStartNodes[1])
                cr_node_int = self.TravStartNodes[0][cr_node_int]
                print('loope%d'%while_count)
                while_count = while_count + 1
            else:
                cr_node_int = np.random.choice(list(self.init_str2int_dict.keys()))
                print('break%d'%while_count)
                #break
        cr_node_int = self.init_str2int_dict[cr_node_int]
        # for cr_node_int in init_hour_idx:
        #     if self.NodesStr[cr_node_int][0] == 'H':
        #         cr_node_int = cr_node_int
        #         break
        hour = 0
        trav_seq.append(self.NodesStr[cr_node_int])
        while (cr_node_int != None) and (hour < 24) :
            cr_node_int,cr_node_str = self.findNext(cr_node_int,hour,temperature[hour])
            #print('cr_node_int',cr_node_int)
            if cr_node_str != None:
                trav_seq.append(cr_node_str)
                #print(cr_node_str, hour+1)
            hour = hour + 1
            #print(hour)
        while hour < 24:
            #print(hour)
            #print(trav_seq[-1])
            trav_seq.append(trav_seq[-1].split('.')[0]+'.'+str(hour))
            hour = hour + 1
        #print('End of Travelsal.')
        return trav_seq

    def inquirySeqPob(self,input_seq):
        pob_list = []
        init_pob = self.TravStartNodes[1][np.argwhere(self.TravStartNodes[0] == input_seq[0].split('.')[0])]
        pob_list.append(float(init_pob))
        self.value2node = {y: x for x, y in self.NodesValue.items()}

        for i in range(1,len(input_seq)):
            o_index = self.value2node[self.node2index[input_seq[i-1]]]
            d_index = self.value2node[self.node2index[input_seq[i]]]
            #print(input_seq[i-1],input_seq[i])
            cr_pob = self.EdgesPob[(o_index,d_index)]
            pob_list.append(cr_pob)
        return pob_list

    def visGraph(self, temp_graph=[], edge_attr_name = 'EdgePob',save_name='temp'):
        if temp_graph == []:
            temp_graph=self.G
        #net = Network(notebook=True, directed=True)
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
        group_dict = {'H': 'Home', 'W': 'Work', 'O': 'Other'}
        edge_list = list(temp_graph.edges())
        node_str_dict = nx.get_node_attributes(temp_graph, 'KeyPointStr')
        edge_weight_dict = nx.get_edge_attributes(temp_graph, edge_attr_name)  #
        for i in range(len(edge_list)):
            src_id = int(edge_list[i][0])
            dst_id = int(edge_list[i][1])
            src_label = node_str_dict[src_id]
            dst_label = node_str_dict[dst_id]
            w = str(edge_weight_dict[edge_list[i]])
            src_group = group_dict[src_label[0]]
            dst_group = group_dict[dst_label[0]]
            # src_label = src_label[0]+' Hour '+src_label[-2:]
            # dst_label = dst_label[0]+' Hour '+dst_label[-2:]
            net.add_node(src_id, label=src_label, title=src_label, group=src_group)
            net.add_node(dst_id, label=dst_label, title=dst_label, group=dst_group)
            net.add_edge(src_id, dst_id, value=w)
            # export to csv
            # edge_dict_list.append({'uid':uid,'O_node':src_label,'D_node':dst_label,'edge_weight':w})
            # print(src_label)
        # neighbor_map = net.get_adj_list()

        # # add neighbor data to node hover data
        # for node in net.nodes:
        #     node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
        #     node['value'] = len(neighbor_map[node['id']])

        net.show_buttons(filter_=['physics'])  #
        #     net.set_options(
        #     '''
        #     var options = {
        #       "physics": {
        #         "forceAtlas2Based": {
        #           "springLength": 100
        #         },
        #         "minVelocity": 0.75,
        #         "solver": "forceAtlas2Based"
        #       }
        #     }
        #     '''
        #     )
        net.save_graph(save_name +'_graph_view.html')
        return net

#trans node_atrr_str to feature_dim
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    Args:
      y: class vector to be converted into a matrix
          (integers from 0 to num_classes).
      num_classes: total number of classes. If `None`, this would be inferred
        as the (largest number in `y`) + 1.
      dtype: The data type expected by the input. Default: `'float32'`.

    Returns:
      A binary matrix representation of the input. The classes axis is placed
      last.

    Example:

    >>> a = tf.keras.tools.to_categorical([0, 1, 2, 3], num_classes=4)
    >>> a = tf.constant(a, shape=[4, 4])
    >>> print(a)
    tf.Tensor(
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]], shape=(4, 4), dtype=float32)

    >>> b = tf.constant([.9, .04, .03, .03,
    ...                  .3, .45, .15, .13,
    ...                  .04, .01, .94, .05,
    ...                  .12, .21, .5, .17],
    ...                 shape=[4, 4])
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(np.around(loss, 5))
    [0.10536 0.82807 0.1011  1.77196]

    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(np.around(loss, 5))
    [0. 0. 0. 0.]

    Raises:
      Value Error: If input contains string value

    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def nodeStr2Feat(temp_node_str,MODE='ONEHOT'):
    if MODE=='HWOT':
        loc_type_list = np.array(['H','W','O'])
        new_node_attrs = []
        if type(temp_node_str)== str or type(temp_node_str)== np.str:
            temp_str = temp_node_str
            loc_type = np.argwhere(loc_type_list==temp_str[0])
            loc_type = to_categorical(range(3))[int(loc_type)].astype(float)
            candi_type = float(temp_str[2]) / 10
            hour_type = float(temp_str.split('.')[-1])/23
            node_attr_feat = np.append(loc_type, [candi_type,hour_type])#,temp_str[0]])
            new_node_attrs.append(node_attr_feat)
            return np.array(new_node_attrs)
        for i in range(len(temp_node_str)):
            temp_str = temp_node_str[i]
            loc_type = np.argwhere(loc_type_list==temp_str[0])
            loc_type = to_categorical(range(3))[int(loc_type)].astype(float)
            candi_type = float(temp_str[2]) / 10
            hour_type = float(temp_str.split('.')[-1])/23
            node_attr_feat = np.append(loc_type, [candi_type,hour_type])#,temp_str[0]])
            new_node_attrs.append(node_attr_feat)
        return np.array(new_node_attrs)
    else:
        loc_type_list = np.array(['H','W','O'])
        new_node_attrs = []
        if type(temp_node_str)== str or type(temp_node_str)== np.str:
            temp_str = temp_node_str
            loc_type = np.argwhere(loc_type_list==temp_str[0])
            loc_type = to_categorical(range(3))[int(loc_type)].astype(float)
            candi_type = int(temp_str[2])
            if candi_type <8:
                candi_type = to_categorical(range(8))[candi_type].astype(float)
            else:
                candi_type = to_categorical(range(8))[7].astype(float)
            hour_type = float(temp_str.split('.')[-1])/23
            node_attr_feat = np.append(np.append(loc_type, candi_type),hour_type)#,temp_str[0]])
            new_node_attrs.append(node_attr_feat)
            return np.array(new_node_attrs)
        for i in range(len(temp_node_str)):
            temp_str = temp_node_str[i]
            loc_type = np.argwhere(loc_type_list==temp_str[0])
            loc_type = to_categorical(range(3))[int(loc_type)].astype(float)
            candi_type = int(temp_str[2])
            if candi_type <8:
                candi_type = to_categorical(range(8))[candi_type].astype(float)
            else:
                candi_type = to_categorical(range(8))[7].astype(float)
            hour_type = float(temp_str.split('.')[-1])/23
            node_attr_feat = np.append(np.append(loc_type, candi_type),hour_type)#,temp_str[0]])
            new_node_attrs.append(node_attr_feat)
        return np.array(new_node_attrs)


def crossEntropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = - np.sum(targets * np.log(predictions)) / N
    return ce


# Function: calcu the cross entropy of two array
def crossCrossEntropy(true_seqs_emds, trav_seqs_emds):
    if len(true_seqs_emds) != len(trav_seqs_emds):
        print('Please give equal amount of seqs.')
    ce_errors = np.zeros([len(true_seqs_emds), len(trav_seqs_emds)])
    for i in range(len(true_seqs_emds)):
        for j in range(len(trav_seqs_emds)):
            t_seq = true_seqs_emds[i]
            s_seq = trav_seqs_emds[j]
            ce_error = crossEntropy(s_seq, t_seq)
            ce_errors[i, j] = ce_error
    return ce_errors


def getNpSmallestK(arr, k):
    if k>1:
        idx = np.argpartition(arr.ravel(), k)
        return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    else:
        return 0
    # if you want it in a list of indices . . .
    # return np.array(np.unravel_index(idx, arr.shape))[:, range(k)].transpose().tolist()

def getFreqSeq(seqs,EXPORT_MODE='Pob'):
    str_sqes = []
    for seq in seqs:
        seq_str = ''
        for node in seq:
            seq_str = seq_str + '*' + node
        str_sqes.append(seq_str)
    unique, counts = np.unique(str_sqes, return_counts=True)
    unique = [x.split('*')[1:] for x in unique]
    if EXPORT_MODE =='Pob':
        return unique,counts/np.sum(counts)
    else:
        return unique, counts

def getMostFreqSeq(seqs,k=100):
    str_sqes = []
    for seq in seqs:
        seq_str = ''
        for node in seq:
            seq_str = seq_str + '*' + node
        str_sqes.append(seq_str)
    unique, counts = np.unique(str_sqes, return_counts=True)
    if k > len(unique):
        k = len(unique)
    most_freq_k = unique[np.argsort(-counts)[:k]]
    most_freq_k = [x.split('*')[1:] for x in most_freq_k]
    return most_freq_k,k

from pyvis.network import Network
#Visulization with graphVis
def graphVisulize(temp_graph,node_attr_name='KeyPointStr',edge_attr_name='EdgePob',save_name=''):
    net = Network(notebook = True,directed=True)
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    group_dict = {'H':'Home','W':'Work','O':'Other'}
    edge_list = list(temp_graph.edges())
    node_str_dict = nx.get_node_attributes(temp_graph,node_attr_name)
    edge_weight_dict = nx.get_edge_attributes(temp_graph,edge_attr_name) #
    for i in range(len(edge_list)):
        src_id = int(edge_list[i][0])
        dst_id = int(edge_list[i][1])
        src_label = node_str_dict[src_id]
        dst_label = node_str_dict[dst_id]
        w = str(edge_weight_dict[edge_list[i]])
        src_group = group_dict[src_label[0]]
        dst_group = group_dict[dst_label[0]]
        #src_label = src_label[0]+' Hour '+src_label[-2:]
        #dst_label = dst_label[0]+' Hour '+dst_label[-2:]
        net.add_node(src_id, label = src_label,title = src_label,group=src_group)
        net.add_node(dst_id, label = dst_label,title = dst_label,group=dst_group)
        net.add_edge(src_id, dst_id, value=w, title=str(w))
        #export to csv
        #edge_dict_list.append({'uid':uid,'O_node':src_label,'D_node':dst_label,'edge_weight':w})
        #print(src_label)
    #neighbor_map = net.get_adj_list()

    # # add neighbor data to node hover data
    # for node in net.nodes:
    #     node['title'] += ' Neighbors:<br>' + '<br>'.join(neighbor_map[node['id']])
    #     node['value'] = len(neighbor_map[node['id']])

    net.show_buttons(filter_=['physics']) #
#     net.set_options(
#     '''
#     var options = {
#       "physics": {
#         "forceAtlas2Based": {
#           "springLength": 100
#         },
#         "minVelocity": 0.75,
#         "solver": "forceAtlas2Based"
#       }
#     }
#     '''
#     )
    net.show(save_name + '_user.html')
    return net


# save_name = 'true'
# i= 4
# graphVisulize(save_name,i,true_graphs[i],'EdgePob')