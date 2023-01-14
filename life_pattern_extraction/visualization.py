import networkx as nx
from pyvis.network import Network


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