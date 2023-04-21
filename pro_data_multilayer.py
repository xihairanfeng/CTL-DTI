import re
import os
import networkx as nx
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder

def get_nodes_edges(filepath):
    readfile = open(filepath)
    all_nodes = []
    all_edges = []
    for line in readfile:
        edge = line.split()
        if int(edge[0]) not in all_nodes:
            all_nodes.append(int(edge[0]))
        if int(edge[1]) not in all_nodes:
            all_nodes.append(int(edge[1]))
        all_edges.append([int(edge[0]), int(edge[1])])
    readfile.close()
    return all_nodes, all_edges

def get_graph(nodes, edges):
    print('get_graph_nodes_number',len(nodes))
    print('get_graph_edges_number',len(edges))
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    return graph

def load_data(filepath):
    all_nodes, all_edges = get_nodes_edges(filepath)
    graph = get_graph(all_nodes, all_edges)
    return graph

def get_edge_embs(graph, savepath):
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit()
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    edges_kv = edges_embs.as_keyed_vectors()
    edges_kv.save_word2vec_format(savepath)

def get_node_index(str):
    result = re.findall(r"\d+", str)
    return int(result[0])

def write_info(file, link_label, network_label, edge_infor):
    edge_infor[0] = link_label
    edge_infor[1] = network_label
    file.write(edge_infor[0])
    for infor in edge_infor[1:]:
        file.write(',' + infor)
    file.write('\n')

def get_labels(graph, temppath, writepath, network_label):
    readfile = open(temppath)
    read_infor = readfile.readline()
    print("edge_embedding_vector, number * dimension: ", read_infor)

    writefile = open(writepath, 'a', encoding='utf-8')

    for line in readfile:
        edge_infor = line.split()
        letfNode = get_node_index(edge_infor[0])
        rightNode = get_node_index(edge_infor[1])
        if letfNode == rightNode or graph.has_edge(letfNode, rightNode) or graph.has_edge(rightNode, letfNode):
            write_info(writefile, '1', str(network_label), edge_infor)
        else:
            write_info(writefile, '0', str(network_label), edge_infor)

    readfile.close()
    writefile.close()

if __name__ == '__main__':
    dataset = 'durg'
    path = './dataset/'
    path_list = os.listdir(path)

    network_total = len(path_list)
    print(dataset, 'network_total', network_total)
    temppath = './node2vec_multilayer/' + dataset + '_temp.txt'

    for i in range(network_total):
        network_layer_list = [i for i in range(network_total)]
        network_label = network_layer_list[i]
        print('The target layler is: ', network_label)
        target_layer_path = './dataset/' + dataset + '_' + str(network_label) + '.txt'
        write_target_file = './node2vec_multilayer/' + 'network_' + str(network_label) + '_target.txt'
        write_auxiliary_file = './node2vec_multilayer/' + 'network_' + str(network_label) + '_auxiliary.txt'

        graph = load_data(target_layer_path)
        get_edge_embs(graph, temppath)
        get_labels(graph, temppath, write_target_file, network_label)

        network_layer_list.pop(i)
        for layer in network_layer_list:
            network_label = layer
            print('The auxiliary layer is:', network_label)
            auxiliary_network = './dataset/' + dataset + '_' + str(network_label) + '.txt'
            graph = load_data(auxiliary_network)
            get_edge_embs(graph, temppath)
            get_labels(graph, temppath, write_auxiliary_file, network_label)

    if os.path.exists(temppath):
        os.remove(temppath)
    print('network numbers:', network_label)
