import pickle
import random
import numpy as np
from tqdm import tqdm

def loadFromPickle(fname):
    '''
        read hin from pickle
    '''
    network = pickle.load(open(fname, 'rb+'))
    return network


def generate_train_data():
    sub = loadFromPickle('../data/DBpedia/hin.pickle')
    print('node number is:', sub.node_num())
    print('edge number is:', sub.edge_count)
    x = sub.sample_subgraph()

    print('node number is:', x.node_num())
    print('edge number is:', x.edge_count)
    x.dump_to_file('../data/DBpedia/train_not_remove_edges.pickle')

def remove_edges_from_train_data(path):
    #  remove 10% edges
    # '../data/DBpedia/hin.pickle'
    sub = loadFromPickle(path)

    print('node number is:', sub.node_num())
    print('edge number is:', sub.edge_count)
    sub.random_remove_edges()

    print('node number is:', sub.node_num())
    print('edge number is:',sub.edge_count)
    sub.dump_to_file('../data/DBpedia/train.pickle')
def gennerate_id_class_min_max():
    # f= loadFromPickle('../../dataset/dbp/hin.pickle')
    f= loadFromPickle('./data/DBpedia/train_not_remove_edges.pickle')

    # print(f.node_class[1])
    dic={}
    class_ids=sorted(f.node_class.keys())
    for class_id in class_ids:
        ids=f.node_class[class_id]
        max_id=max(ids)
        min_id=min(ids)
        for i in ids:
            dic[i]=[class_id,min_id,max_id]
    print(len(dic))
    '''
    dic:
    {104832: [0, 17602, 191082]}
    {node_id:[node_type_id,min_node_id,max_node_id]}
    '''

    nes_ids=sorted(dic.keys())
    print(len(nes_ids))
    for node_id in nes_ids:
        with open('id_to_type_and_range','a') as ff:
            line=str(dic[node_id][0])+'\t' +str(dic[node_id][1])+'\t'+str(dic[node_id][2])
            # print(node_id,'node_id:',line)
            ff.write(line+'\n')

def gennerate_walk_file_train():
    g = loadFromPickle('../data/DBpedia/train.pickle')
    g.generate_random_walks(10, 100)

def gennerate_walk_file_test():
    g = loadFromPickle('./data/DBpedia/train_not_remove_edges.pickle')
    g.generate_random_walks(10, 100)

def gennerate_test_data(seed=None):
    f = loadFromPickle('../../dataset/dbp/s.pickle')
    # node_class=f.node_class  #{node_class: set([node_id])}
    num_repeat=1
    # pos_s_pairs-0
    # for length in [1,2,3,4]:
    #     f.generate_random_walks(num_repeat, length)

    # neg_s_pairs
    dic={}
    class_ids=sorted(f.node_class.keys())
    for class_id in class_ids:
        ids=f.node_class[class_id]
        for i in ids:
            dic[i]=class_id
    print(len(dic))
    # dic {node_id:node_class_id}
    lines=open('pos_s_pairs-3','r').readlines()
    for line in lines:
        sp=[int(x) for x in line.split('\t')]
        print(sp)
        # 随机替换一个节点 替换后的节点还是是同一种类型的节点
        h_node=sp[0]
        t_node=sp[-1]
        fla= random.randint(1,2)
        if fla ==1:
            # 替换h
            class_h=dic[h_node]
            nodes=list(f.node_class[class_h])
            m = 0
            while 1:
                # 去掉自己
                index=random.randint(0,len(nodes)-1)
                # 替换成这个
                temp=nodes[index]
                m += 1
                if temp - h_node !=0:
                    sp[0]=temp
                    break
                else:
                    if m > 3:
                        print('nodes not much:',nodes)
                        break
        else:
            # 替换t
            class_t=dic[t_node]
            nodes=list(f.node_class[class_t])
            m = 0
            while 1:
                index=random.randint(0,len(nodes)-1)
                # 替换成这个
                temp=nodes[index]
                if temp - h_node != 0:
                    sp[-1]=temp
                    break
                else:
                    if m > 3:
                        print('nodes not much:',nodes)
                        break
        ll='\t'.join([str(y) for y in sp])
        with open('net_s_pairs-3','a') as ne:
            ne.write(ll+'\n')

def load_lookup_from_file(file):
    '''
    load the lookup table from file
    '''
    result = []
    with open(file) as f:
        for line in f:
            result.append(list(map(lambda x:int(x),line.split())))
    return np.array(result)


def generate_corrupt_trip(trip,f_train):
    # f = loadFromPickle('../data/DBpedia/train.pickle')
    f = f_train
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
                print('only self ?:', 'm:', m)
                if m > 3:
                    print('nodes not much:',nodes)
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
                print('only self:', 'm:', m)
                if m > 3:
                    print('nodes not much:',nodes)
                    break
    # print('corrupt trip:', sp)
    ll = '\t'.join([str(y) for y in sp])
    with open('../data/DBpedia/corrupt', 'a') as f:
        f.write(ll + '\n')
    # return sp


def generate_trip(path_len,f_walk,f_train):
    # f = open('../data/DBpedia/walk_10_100','r')
    f = f_walk
    i = 1
    for line in f.readlines():
        # if i % 10000 == 0:
        #     print('10000,success',i)
        # i += 1
        lis = list(map(int, line.split('\t')))
        nodes = lis[::2]
        edges = lis[1::2]
        for index in tqdm(range(101 - path_len)):
            head = str(nodes[index])+'\t'
            for i in range(path_len):
                head += str(edges[index+i]) + '\t'
                head += str(nodes[index+i+1]) + '\t'
            trip = head + '\n'
            # print(trip,type(head),head)
            generate_corrupt_trip(head,f_train)
            # break
            with open('../data/DBpedia/valid', 'a') as f:
                f.write(trip)

        # break

def generate_train_trip():
    f_walk = open('../data/DBpedia/walk_10_100', 'r')
    f_train = loadFromPickle('../data/DBpedia/train.pickle')
    for i in range(4,5):
        print('generate meta-path with length of :', i)
        generate_trip(i,f_walk,f_train)

def mearge_data_test(x,y):
    path = '../data/DBpedia/'
    for i in x:
        i = path+i
        f = open(i, 'r').readlines()
        for line in f:
            with open('../data/DBpedia/test_pos', 'a') as ff:
                ff.write(line)
    for j in y:
        j = path+j
        f = open(j, 'r').readlines()
        for line in f:
            with open('../data/DBpedia/test_neg', 'a') as ff:
                ff.write(line)

def mearge_data_train(x,y):
    path = '../data/DBpedia/'
    for i in x:
        i = path+i
        f = open(i, 'r').readlines()
        for line in f:
            with open('../data/DBpedia/train_pos', 'a') as ff:
                ff.write(line)
    for j in y:
        j = path+j
        f = open(j, 'r').readlines()
        for line in f:
            with open('../data/DBpedia/train_neg', 'a') as ff:
                ff.write(line)

if __name__ == '__main__':
    # x=['pos_s_pairs-0','pos_s_pairs-1','pos_s_pairs-2','pos_s_pairs-3']
    # y=['neg_s_pairs-0','neg_s_pairs-1','neg_s_pairs-2','neg_s_pairs-3']
    x=['valid_1','valid_2','valid3','valid']
    y=['corrupt_1','corrupt_2','corrupt3','corrupt']
    mearge_data_train(x,y)
    # pass
    # first
    # generate_train_data()
    # second
    # remove_edges_from_train_data('../data/DBpedia/train_not_remove_edges.pickle')
    # random walks 10 times for each node with the length of 100
    # gennerate_walk_file_train()

    # generate_train_trip()
    # generate_trip(2)
    # generate_corrupt_trip('10325	15	10326')

    # generate_trip(1)

    # f=loadFromPickle('../data/DBpedia/train_not_remove_edges.pickle')
    # # print(f)
    # print('node number is:',f.node_num())
    # print('node class number is:',len(f.node_class.keys()))
    # print('edge number is:',f.edge_count)
    # print('edge class number is:',len(f.edge_class_id.keys()))
    # print(f.node_class[1])


    # before remove edges
    # node
    # number is: 170742
    # class number is: 357
    # edge
    # number is: 336934
    # class number is: 976

    # train data  remove 10% edges
    # node
    # number is: 170742
    # class number is: 357
    # edge
    # number is: 303240
    # class number is: 976