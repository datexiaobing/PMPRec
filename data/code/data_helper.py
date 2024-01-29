import random, os,time
import numpy as np
import networkx as nx
import threading
from multiprocessing import Process

# 5 node type 8 edge type
class DataHelper():
    def __init__(self,):
        self.m_a = self.get_data_lines('../movie_actor.dat')
        self.m_d = self.get_data_lines('../movie_director.dat')
        self.m_t = self.get_data_lines('../movie_type.dat')
        # self.u_g = self.get_data_lines('../user_book.dat')
        self.u_m = self.get_data_lines('../user_movie.dat')
        self.u_u = self.get_data_lines('../user_user.dat')
        self.rates = [0.2, 0.4, 0.6, 0.8]
        self.edge_id_dic_to_id = {
            'md': '0',
            'dm': '1',
            'ma': '2',
            'am': '3',
            'mt': '4',
            'tm': '5',
            'um': '6',
            'mu': '7',
        }
        self.meta_path_list_3_list = [[('u','m','u')],[('m','t','m')],[('m','d','m')],[('m','a','m')],[('m','u','m')]]
        self.meta_path_list_5_list =[[('u', 'm', 't','m','u')],[('u', 'm', 'a','m','u')],[('u', 'm', 'd','m','u')]]
        self.mp3_rate = 0.3
        self.mp5_rate = 0.08
        self.dic_new_id = {}  # {'m0':0,'a0':1111} old_id --> new id
        self.new_id_class = {}  # {id:class}  new_id --> node class a,m,d,t,g,u
        self.class_nodes_id = {}  # [' m':['ids']]--> class id => node id list
        self.dic_new_id_to_old_id = {}
        self.node_id_to_class = []

        self.generate_new_id()

    def get_data_lines(self, file_path):
        return open(file_path, 'r').readlines()

    def generate_new_id(self):
        movies_id = set([])
        actors_id = set([])
        directors_id = set([])
        type_id = set([])
        user_id = set([])
        # group_id = set([])

        for line in self.m_a:
            token = line.strip('\n').split('\t')
            movies_id.add(token[0])
            actors_id.add(token[1])
        for line in self.m_d:
            token = line.strip('\n').split('\t')
            movies_id.add(token[0])
            directors_id.add(token[1])
        for line in self.m_t:
            token = line.strip('\n').split('\t')
            movies_id.add(token[0])
            type_id.add(token[1])
        for line in self.u_m:
            token = line.strip('\n').split('\t')
            user_id.add(token[0])
            movies_id.add(token[1])
        for line in self.u_u:
            token = line.strip('\n').split('\t')
            user_id.add(token[0])
            user_id.add(token[1])
        print('movies:', len(movies_id))
        print('actors:', len(actors_id))
        print('directors:', len(directors_id))
        print('users:', len(user_id))
        print('types:', len(type_id))
        print('node number is:',len(movies_id)+len(actors_id)+len(directors_id)+len(user_id)+len(type_id))
        # 对set进行排序，保证每次加载数据，old_id => new_id 不会变动
        movies_id = self.sort_list(movies_id)
        actors_id = self.sort_list(actors_id)
        directors_id = self.sort_list(directors_id)
        user_id = self.sort_list(user_id)
        type_id = self.sort_list(type_id)

        # 重新编码新的id
        dic_new_id = {}  # {'m0':0,'a0':1111} 原来id--> new id
        new_id_class = {}  # {id:class}  new)id --> node class a,m,d,t,g,u
        class_nodes_id = {}  # ['m':['movies id]]
        dic_new_id_to_old_id = {}
        node_id_to_class = []
        # 重新编码id
        index = 0
        m_list = []
        a_list = []
        d_list = []
        t_list = []
        u_list = []
        # g_list = []
        for m in movies_id:
            dic_new_id_to_old_id[str(index)] = m
            node_id_to_class.append('0')
            dic_new_id['m' + m] = index
            new_id_class[str(index)] = 'm'
            m_list.append(str(index))
            index += 1
        for a in actors_id:
            dic_new_id_to_old_id[str(index)] = a
            node_id_to_class.append('1')
            dic_new_id['a' + a] = index
            new_id_class[str(index)] = 'a'
            a_list.append(str(index))
            index += 1
        for d in directors_id:
            dic_new_id_to_old_id[str(index)] = d
            node_id_to_class.append('2')
            dic_new_id['d' + d] = index
            new_id_class[str(index)] = 'd'
            d_list.append(str(index))
            index += 1
        for d in user_id:
            dic_new_id_to_old_id[str(index)] = d
            node_id_to_class.append('3')
            dic_new_id['u' + d] = index
            new_id_class[str(index)] = 'u'
            u_list.append(str(index))
            index += 1
        for d in type_id:
            dic_new_id_to_old_id[str(index)] = d
            node_id_to_class.append('4')
            dic_new_id['t' + d] = index
            new_id_class[str(index)] = 't'
            t_list.append(str(index))
            index += 1

        class_nodes_id['m'] = m_list
        class_nodes_id['a'] = a_list
        class_nodes_id['d'] = d_list
        class_nodes_id['u'] = u_list
        class_nodes_id['t'] = t_list
        # class_nodes_id['g'] = g_list

        self.dic_new_id = dic_new_id
        self.new_id_class = new_id_class
        self.class_nodes_id = class_nodes_id
        self.dic_new_id_to_old_id = dic_new_id_to_old_id
        self.node_id_to_class = node_id_to_class

    def sort_list(self, books_id):
        books_id = [int(i) for i in books_id]
        books_id.sort()
        return [str(i) for i in books_id]

    def generate_node_id_to_class_file(self):
        path = r'../node_id_to_class.txt'
        if os.path.exists(path):
            print(path,' file exists,no need to generate again !')
            return
        for nodes_id in self.node_id_to_class:
            with open(path, 'a') as f:
                f.write(nodes_id + '\n')
        print('new_id_to_class file done')

    def generate_new_id_ubFile(self):
        path = '../new_u_m.txt'
        if os.path.exists(path):
            print(path,'file exists,no need to generate again !')
            return
        for line in self.u_m:
            f = line.strip('\n').split('\t')
            u_new = self.dic_new_id['u' + f[0]]
            b_new = self.dic_new_id['m' + f[1]]
            s = str(u_new) + ' ' + str(b_new) + ' ' + f[2] + '\n'
            with open(path, 'a') as ff:
                ff.write(s)
        print('new u movie file done .please split this file for next train and test stage')

    def generate_all_edges_file(self):
        path = '../movie_edges.txt'
        if os.path.exists(path):
            print('all_edges file exists,no need to generate again !')
            return
        with open(path, 'a') as imf:
            for line in self.m_a:
                token = line.strip('\n').split('\t')
                new_token_0 = str(self.dic_new_id['m' + token[0]])
                new_token_1 = str(self.dic_new_id['a' + token[1]])
                imf.write(new_token_0 + '\t' + new_token_1 + '\n')
            for line in self.m_d:
                token = line.strip('\n').split('\t')
                new_token_0 = str(self.dic_new_id['m' + token[0]])
                new_token_1 = str(self.dic_new_id['d' + token[1]])
                imf.write(new_token_0 + '\t' + new_token_1 + '\n')
            for line in self.m_t:
                token = line.strip('\n').split('\t')
                new_token_0 = str(self.dic_new_id['m' + token[0]])
                new_token_1 = str(self.dic_new_id['t' + token[1]])
                imf.write(new_token_0 + '\t' + new_token_1 + '\n')
            for line in self.u_m:
                token = line.strip('\n').split('\t')
                new_token_0 = str(self.dic_new_id['u' + token[0]])
                new_token_1 = str(self.dic_new_id['m' + token[1]])
                imf.write(new_token_0 + '\t' + new_token_1 + '\n')
            # for line in self.u_u:
            #     token = line.strip('\n').split('\t')
            #     new_token_0 = str(self.dic_new_id['u' + token[0]])
            #     new_token_1 = str(self.dic_new_id['u' + token[1]])
            #     imf.write(new_token_0 + '\t' + new_token_1 + '\n')
        print('generate movie_all_edges done !')

    def split_train_test_data(self,file_path, test_data_rate, save_path_train, save_path_test):

        a_lis = np.loadtxt(file_path, dtype="str")
        a_length = len(a_lis)
        test_data_len = int(a_length * test_data_rate)
        print('data length:', a_length, 'test len', test_data_len)

        # 生成随机下标，不重复的,测试机
        random_index = np.random.choice(a_length, test_data_len, replace=False)
        print('generate test index done !')

        np_a = np.array(a_lis)

        test_arr = np_a[random_index]
        print("======saving test data======")
        np.savetxt(save_path_test, test_arr, fmt="%s")

        train_arr = np.delete(np_a, random_index, axis=0)
        print("======saving train data======")
        np.savetxt(save_path_train, train_arr, fmt="%s")

        print(file_path, 'done!')

    def generate_train_test_data(self,):
        base_dir = r'../'
        file = 'new_u_m.txt'
        file_path = base_dir + file
        # 划分测试集
        rates = self.rates
        for test_data_rate in rates:
            # test_data_rate = 0.2
            train_rate = str(int(10 - test_data_rate * 10))
            save_path_base = r"../rate/" + train_rate + '/'
            save_path_train = save_path_base + file + '.' + train_rate
            if os.path.exists(save_path_train):
                print(save_path_train,' file exists,no need to generate again !')
                continue
            save_path_test = save_path_base + 'test_' + file + '.' + train_rate
            if not os.path.exists(save_path_base):
                os.makedirs(save_path_base)
            self.split_train_test_data(file_path, test_data_rate, save_path_train, save_path_test)

    def get_herec_split_data(self,herec_file_name, file):
        with open(herec_file_name, 'a') as ff:
            for line in file:
                l = line.strip().split(' ')
                token_0 = self.dic_new_id_to_old_id[l[0]]
                token_1 = self.dic_new_id_to_old_id[l[1]]
                token_2 = int(float(l[2]))
                new_str = token_0 + '\t' + token_1 + '\t' + str(token_2)
                ff.write(new_str + '\n')
        print(herec_file_name,'done!')

    def generate_train_test_data_for_herec(self,):
        rates = [int(i*10)  for i in self.rates]
        for rate in rates:
            f_8 = '../rate/' + str(rate) + '/new_u_m.txt.' + str(rate)
            u_b_8 = open(f_8, 'r').readlines()
            test_f_8 = '../rate/' + str(rate) + '/test_new_u_m.txt.' + str(rate)
            test_u_b_8 = open(test_f_8, 'r').readlines()
            train_f = '../rate/' + str(rate) + '/um_0.' + str(rate) + '.train'
            test_f = '../rate/' + str(rate) + '/um_0.' + str(rate) + '.test'

            self.get_herec_split_data(train_f,u_b_8)
            self.get_herec_split_data(test_f, test_u_b_8)

    def generate_train_edges_file(self):
        rates = [int(i * 10) for i in self.rates]
        for rate in rates:
            f_8 = '../rate/' + str(rate) + '/new_u_m.txt.' + str(rate)
            u_b_8 = open(f_8, 'r').readlines()
            path = '../metaPath/movie_edges_0.' + str(rate) + '.txt'
            if os.path.exists(path):
                print(path,' file exists,no need to generate again !')
                continue
            with open(path, 'a') as imf:
                for line in self.m_a:
                    token = line.strip('\n').split('\t')
                    new_token_0 = str(self.dic_new_id['m' + token[0]])
                    new_token_1 = str(self.dic_new_id['a' + token[1]])
                    imf.write(new_token_0 + '\t' + new_token_1 + '\n')
                for line in self.m_d:
                    token = line.strip('\n').split('\t')
                    new_token_0 = str(self.dic_new_id['m' + token[0]])
                    new_token_1 = str(self.dic_new_id['d' + token[1]])
                    imf.write(new_token_0 + '\t' + new_token_1 + '\n')
                for line in self.m_t:
                    token = line.strip('\n').split('\t')
                    new_token_0 = str(self.dic_new_id['m' + token[0]])
                    new_token_1 = str(self.dic_new_id['t' + token[1]])
                    imf.write(new_token_0 + '\t' + new_token_1 + '\n')
                for line in u_b_8:
                    token = line.strip('\n').split(' ')
                    # new_token_0 = str(dic_new_id['u' + token[0]])
                    # new_token_1 = str(dic_new_id['m' + token[1]])
                    new_token_0 = str(token[0])
                    new_token_1 = str(token[1])
                    imf.write(new_token_0 + '\t' + new_token_1 + '\n')
                # for line in self.u_u:
                #     token = line.strip('\n').split('\t')
                #     new_token_0 = str(self.dic_new_id['u' + token[0]])
                #     new_token_1 = str(self.dic_new_id['u' + token[1]])
                #     imf.write(new_token_0 + '\t' + new_token_1 + '\n')
            print('generate train_edges done !', rate)

    def get_next_neighbors(self, G, this_node, mp, new_id_class, rate, only_one=False):
        neighbors = [n for n in G.neighbors(this_node) if new_id_class[n] == mp]
        if len(neighbors) < 2:
            return neighbors
        if only_one:
            return random.sample(neighbors, 1)

        # 去下一个节点邻居的采样率
        num_neighbors_to_choose = int(len(neighbors) * rate)
        if num_neighbors_to_choose < 1:
            num_neighbors_to_choose = 1
        # print(num_neighbors_to_choose)
        next_node = random.sample(neighbors, num_neighbors_to_choose)
        return next_node

    def get_next_pool_mp3(self,G, this_node, meta_path, new_id_class):
        dic = {}
        for ne in self.get_next_neighbors(G, this_node, meta_path[1], new_id_class, rate=self.mp3_rate):
            lis = []
            # 对每一个ne计算ne满足条件的邻居节点
            neibor_next = self.get_next_neighbors(G, ne, meta_path[2], new_id_class, rate=self.mp3_rate)
            for ne_next in neibor_next:
                lis.append(ne_next)
            dic[ne] = lis
        return dic

    def get_next_pool_mp5(self,G, this_node, meta_path, new_id_class):
        dic = {}
        for ne in self.get_next_neighbors(G, this_node, meta_path[1], new_id_class,rate=self.mp5_rate):
            neibor_next = self.get_next_neighbors(G, ne, meta_path[2], new_id_class,rate=self.mp5_rate)
            dic_1 = {}
            for ne_next in neibor_next:
                dic_2 = {}
                for ne_next_1 in self.get_next_neighbors(G, ne_next, meta_path[3], new_id_class,rate=self.mp5_rate):
                    lis = []
                    for ne_next_2 in self.get_next_neighbors(G, ne_next_1, meta_path[4], new_id_class,rate=self.mp5_rate):
                        lis.append(ne_next_2)
                    dic_2[ne_next_1] = lis
                dic_1[ne_next] = dic_2
            dic[ne] = dic_1
        return dic

    # mete-path leng is 3
    def get_meta_path_3(self, G, meta_path_list,rate,save_path_pos, save_path_neg):
        for meta_path in meta_path_list:
            print(meta_path, ':----**-- starting ---**---')
            file_name = ''
            for mm in meta_path:
                file_name += mm
            for node in G.nodes:
                # 遍历所有节点,深度遍历
                node_type = self.new_id_class[node]
                if node_type == meta_path[0]:
                    next_pool = self.get_next_pool_mp3(G, node, meta_path, self.new_id_class)
                    for k, v in next_pool.items():
                        if v:
                            for n_node in v:
                                edge_1 = self.edge_id_dic_to_id[self.new_id_class[node] + self.new_id_class[k]]
                                edge_2 = self.edge_id_dic_to_id[self.new_id_class[k] + self.new_id_class[n_node]]
                                mp_node_list = [node, edge_1, k, edge_2, n_node]
                                neg_file_name = save_path_neg + 'neg_' + file_name + '_' + str(rate) + '.txt'
                                self.neg_sample(mp_node_list,neg_file_name)
                                stt = '\t'.join(mp_node_list) + '\n'
                                with open(save_path_pos + file_name + '_' + str(rate) + '.txt',
                                          'a') as ff:
                                    ff.write(stt)
        print(meta_path,' end !')

    # mete-path leng is 5
    def get_meta_path_5(self, G, meta_path_list, rate,save_path_pos, save_path_neg):
        for meta_path in meta_path_list:
            print(meta_path, ':----**-- starting ---**---')
            if 't' in meta_path:
                self.mp5_rate = 0.01
            else:
                self.mp5_rate = 0.08
            file_name = ''
            for mm in meta_path:
                file_name += mm
            for node in G.nodes:
                node_type = self.new_id_class[node]
                if node_type == meta_path[0]:
                    next_pool = self.get_next_pool_mp5(G, node, meta_path, self.new_id_class)
                    for k1, v1 in next_pool.items():
                        edge1 = self.edge_id_dic_to_id[self.new_id_class[node] + self.new_id_class[k1]]
                        for k2, v2 in v1.items():
                            edge2 =self.edge_id_dic_to_id[self.new_id_class[k1] + self.new_id_class[k2]]
                            for k3, v3 in v2.items():
                                edge3 = self.edge_id_dic_to_id[self.new_id_class[k2] + self.new_id_class[k3]]
                                if v3:
                                    # 最后一个节点不为空时，才生成meta-path，
                                    for k4 in v3:
                                        # 如果去环，这里就过滤掉，k4 == node 的节点
                                        edge4 = self.edge_id_dic_to_id[self.new_id_class[k3] + self.new_id_class[k4]]
                                        mp_node_list = [node, edge1, k1, edge2, k2, edge3, k3, edge4, k4]
                                        # negative sampling
                                        neg_file_name = save_path_neg + 'neg_' + file_name + '_' + str(
                                            rate) + '.txt'
                                        self.neg_sample(mp_node_list, neg_file_name)
                                        stt = '\t'.join(mp_node_list) + '\n'
                                        # print(stt)
                                        with open(
                                                save_path_pos + file_name + '_' + str(rate) + '.txt',
                                                'a') as ff:
                                            ff.write(stt)
        print(meta_path, ' end !')

    def generate_meta_path_3(self, save_base_path, train_count):
        rates = [int(i * 10) for i in self.rates]
        ps = []
        for rate in rates:
            # save_path_pos = '../metaPath/' + str(rate) + '/pos/'
            save_path_pos = save_base_path + str(rate) + '/' + str(train_count) + '/pos/'
            if not os.path.exists(save_path_pos):
                os.makedirs(save_path_pos)
            save_path_neg = save_base_path + str(rate) + '/' + str(train_count) + '/neg/'
            if not os.path.exists(save_path_neg):
                os.makedirs(save_path_neg)
            G_all = nx.read_edgelist('../movie_edges.txt', delimiter='\t', create_using=nx.Graph())
            print('all node numbers:', len(G_all.nodes), 'all  edges numbers:', len(G_all.edges))

            G = nx.read_edgelist('../metaPath/movie_edges_0.' + str(rate) + '.txt', delimiter='\t', create_using=nx.Graph())
            print('rate:',rate,'node numbers:', len(G.nodes), 'edges numbers:', len(G.edges))
            # 创建子进程实例
            for m_list in self.meta_path_list_3_list:
                # t = threading.Thread(target=self.get_meta_path_3, args=(G,m_list,rate,))
                # t.start()
                # time.sleep(1)
                p = Process(target=self.get_meta_path_3, args=(G,m_list,rate,save_path_pos,save_path_neg))
                ps.append(p)
        # 开启进程
        for i in range(len(ps)):
            ps[i].start()
        # 阻塞进程
        for i in range(len(ps)):
            ps[i].join()

    def generate_meta_path_5(self, save_base_path, train_count):
        rates = [int(i * 10) for i in self.rates]
        ps = []
        for rate in rates:
            # save_path_pos = '../metaPath/' + str(rate) + '/pos/'
            save_path_pos = save_base_path + str(rate) + '/' + str(train_count) + '/pos/'
            if not os.path.exists(save_path_pos):
                os.makedirs(save_path_pos)
            save_path_neg = save_base_path + str(rate) + '/' + str(train_count) + '/neg/'
            if not os.path.exists(save_path_neg):
                os.makedirs(save_path_neg)
            G_all = nx.read_edgelist('../movie_edges.txt', delimiter='\t', create_using=nx.Graph())
            print('all node numbers:', len(G_all.nodes), 'all  edges numbers:', len(G_all.edges))

            G = nx.read_edgelist('../metaPath/movie_edges_0.' + str(rate) + '.txt', delimiter='\t',
                                 create_using=nx.Graph())
            print('rate:',rate,'node numbers:', len(G.nodes), 'edges numbers:', len(G.edges))

            for m_list in self.meta_path_list_5_list:
                # t = threading.Thread(target=self.get_meta_path_5, args=(G, m_list, rate,))
                # t.start()
                # time.sleep(1)
                p = Process(target=self.get_meta_path_5, args=(G,m_list,rate,save_path_pos,save_path_neg))
                ps.append(p)
        # 开启进程
        for i in range(len(ps)):
            ps[i].start()
        # 阻塞进程
        for i in range(len(ps)):
            ps[i].join()

    def neg_sample(self,line_list,file_name):
        last_node = line_list[-1]
        last_node_type = self.new_id_class[last_node]
        # 获取该类型节点的其他节点,并去掉自己
        type_node_list = [i for i in self.class_nodes_id[last_node_type] if i != last_node]
        # 随机选择一个替换last node
        choice = random.choice(type_node_list)
        # 生成新的负样本，并保存
        neg_list = list.copy(line_list)
        neg_list[-1] = choice
        stt = ' '.join(neg_list) + '\n'
        # file_name = 'neg_' + file
        with open(file_name, 'a') as ff:
            ff.write(stt)