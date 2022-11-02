import pandas as pd


def extract_lp_format(support_tree):
    def _extract_lp_format(line):
        return tuple(('{}.{}'.format(str(line.places), str(line.next_places)), int(line.time)))

    container = []

    for i in range(len(support_tree)):
        container.append(_extract_lp_format(support_tree.loc[i, :]))
    return pd.Series(container)


def extract_node2index(support_tree):
    nodes = []

    for i in range(len(support_tree)):
        nodes.append('{}.{}'.format(str(support_tree.loc[i, 'places']), str(support_tree.loc[i, 'time'])))

    container = {}
    nodes_unique = set(nodes)

    for i, item in enumerate(nodes_unique):
        container.update({item: i})

    return container


def extract_index2node(node2index):
    container = {}

    for key in node2index.keys():
        container.update({node2index[key]: key})

    return container
