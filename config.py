import graph.helper
from graph.helper import loadFromPickle
from tqdm import tqdm
import random

model_params = {
    "NODE_NUM": 34842,#样本决定
    "NODE_EMBEDDING_DIM": 100, # node_type 和 node dim 不一定要相等
    "NODE_TYPE_NUM": 5,#样本决定
    "NODE_TYPE_EMBEDDING_DIM": 100,
    "EDGE_NUM": 8,#样本决定 边的类型个数，用于meta-patn embedding
    "EDGE_EMBEDDING_DIM": 200,#node_type 和 node 相加
    'LSTM_HIDDEN_SIZE': 256,
    'ATTENTION_HEAD': 8,
    'FC_HIDDEN_SIZE': 256,
    'mp_encoder': "linear",  # self_Att 时有效
    'node_att': "att_bi",  # self_Att 自注意力考虑各节点的位置信息和权重，att 各节点之间的权重相加,att_bi:加入节点顺序信息
    'pre_switch': 2,  # 2新的mlp 1 able后面加上预测
    'mod_switch': 1,  # 1不变，2返回预测值

}

def generate_corrupt_trip_test(trip, path_len):
    f = loadFromPickle('./data/DBpedia/train_not_remove_edges.pickle')
    # f = f_train
    class_ids = sorted(f.node_class.keys())
    dic = {}  # {node_id:node_class_id}
    for class_id in class_ids:
        ids = f.node_class[class_id]
        for i in ids:
            dic[i] = class_id
    # print(len(dic), 'equal node number',trip)
    # print(trip.strip().split('\t'))
    line = trip.strip()
    sp = [int(x) for x in line.split('\t')]
    # print('vaild trip:',sp)
    # 随机替换一个节点 替换后的节点还是是同一种类型的节点
    h_node = sp[0]
    t_node = sp[-1]
    fla = random.randint(1, 2)
    if fla == 1:
        # 替换h
        class_h = dic[h_node]
        nodes = list(f.node_class[class_h])
        m = 0
        while 1:
            # 去掉自己
            index = random.randint(0, len(nodes) - 1)
            # 替换成这个
            temp = nodes[index]
            if temp - h_node != 0:
                sp[0] = temp
                break
            else:
                m = m + 1
                print('only self:', nodes, 'm:', m)
                if m > 3:
                    print('nodes not much break:',nodes)
                    break
    else:
        # 替换t
        class_t = dic[t_node]
        nodes = list(f.node_class[class_t])
        m = 0
        while 1:
            index = random.randint(0, len(nodes) - 1)
            # 替换成这个
            temp = nodes[index]
            if temp - h_node != 0:
                sp[-1] = temp
                break
            else:
                m = m + 1
                print('only self:', nodes, 'm:', m)
                if m > 3:
                    print('nodes not much:',nodes)
                    break
    # print('corrupt trip:', sp)
    ll = '\t'.join([str(y) for y in sp])
    with open('./data/DBpedia/neg_s_pairs-'+str(path_len - 1), 'a') as f:
        f.write(ll + '\n')
    # return sp


def generate_trip_test(path_len):
    f = open('./data/DBpedia/walk_10_100_test','r')
    # f = f_walk
    j = 0
    for line in f.readlines():
        lis = list(map(int, line.split('\t')))
        nodes = lis[::2]
        edges = lis[1::2]
        for index in tqdm(range(101 - path_len)):
            j += 1
            if j > 1000:
                print('1000:', j)
                return 0
            head = str(nodes[index])+'\t'
            for i in range(path_len):
                head += str(edges[index+i]) + '\t'
                head += str(nodes[index+i+1]) + '\t'
            trip = head + '\n'
            # print(trip,type(head),head)
            generate_corrupt_trip_test(head,path_len)
            # break
            with open('./data/DBpedia/pos_s_pairs-'+str(path_len - 1), 'a') as f:
                f.write(trip)

        # break

# 数据来源于 ./data/hin.pickle
if __name__ == '__main__':
    pass
    # from  graph.helper import *
    # {from_id: {to_id: {edge_class_id: weight},...}}
    # f = loadFromPickle('./data/DBpedia/train.pickle')
    # from_ids = sorted(f.graph.keys())
    # print(len(from_ids),from_ids[0],max(from_ids))
    # print(f.node_class)
    # for from_id in  from_ids:
    #     to_ids =

    # gennerate_id_class_min_max()
    # graph.helper.gennerate_walk_file_test()

    # f = open('./data/DBpedia/walk_10_100_test','r')
    # i = 0
    # for line in f.readlines():
    #     for index in tqdm(range(101 - 1)):
    #         i +=1
    #         print(i)


    # for i in range(1,5):
    #     print('generate meta-path with length of :', i)
    #     generate_trip_test(i)


    # x=[1,2,2,3,3,4]
    # y=set(x)
    # z=y.add(6)
    # print(y)
    # print(z)

