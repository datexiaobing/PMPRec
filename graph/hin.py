import numpy as np
import pickle
import random
import os.path
from tqdm import tqdm


class HINGraph(object):
    def __init__(self):
        self.graph = {}  # {from_id: {to_id: {edge_class_id: weight},...}}
        self.node_id = {}  # {node: node_id}  {4220161： 49114 } 给每个节点分一个id
        self.node_class = {}  # {node_class: set([node_id])}
        self.edge_class_id = {}  # {edge_class: edge_class_id} 给每个边类型分一个id
        self.edge_count = 0
        self.edge_class_matrix = {}
        self.metapath_matrix = {}
        self.next_node_pool = None  # {k: {id_: set(to_ids)}}
        self.reverse_list = None

    def get_node_class(self, graph):
        old_node_class = graph.node_class
        # print(old_node_class[1])
        new_node_class = {}
        index = 0
        for _, v in old_node_class.items():
            new_node_class[index] = v
            index += 1
        graph.node_class = new_node_class



    # 由边生成graph from_node, from_class, to_node, to_class, edge_class, weight
    def add_edge(self, from_node, from_class, to_node, to_class, edge_class, weight=1, remove_self=True):
        # 去自循环 A1 -P1 -A1  这种情况
        if from_node == to_node and remove_self:
            return None
        if from_node not in self.node_id:
            self.node_id[from_node] = len(self.node_id)  # 给新的节点，添加ID（自增）
        from_id = self.node_id[from_node]

        if from_class not in self.node_class:
            self.node_class[from_class] = set()
        self.node_class[from_class].add(from_id)

        if to_node not in self.node_id:
            self.node_id[to_node] = len(self.node_id)
        to_id = self.node_id[to_node]

        if to_class not in self.node_class:
            self.node_class[to_class] = set()
        self.node_class[to_class].add(to_id)

        if edge_class not in self.edge_class_id:
            self.edge_class_id[edge_class] = len(self.edge_class_id)
        edge_id = self.edge_class_id[edge_class]

        if from_id not in self.graph:
            self.graph[from_id] = {}
        self.graph[from_id][to_id] = {edge_id: weight}

        self.edge_count += 1

    def node_num(self):
        return len(self.node_id)

    def edge_num(self):
        return self.edge_count

    def calculate_edge_class_matrix(self):
        dim = self.node_num()
        for i in range(len(self.edge_class_id)):
            self.edge_class_matrix[i] = np.zeros((dim, dim), dtype=int)

        # add each edge to associated matrix
        for from_id in self.graph:
            for to_id in self.graph[from_id]:
                for edge_id, weight in self.graph[from_id][to_id].items():
                    self.edge_class_matrix[edge_id][from_id][to_id] = weight

    def print_matrix(self):
        for edge_class in self.edge_class_matrix:
            print(edge_class, ":")
            print(self.edge_class_matrix[edge_class])
            print()

    def calculate_meta_path_adjacency_matrix(self, meta_path):
        adj_matrix = np.identity(self.node_num())
        for edge in meta_path:
            id_ = self.edge_class_id[edge]
            adj_matrix = np.dot(adj_matrix, self.edge_class_matrix[id_])
        return adj_matrix

    def get_id_from_hin(self, hin):
        self.node_id = hin.node2id
        self.edge_class_id = hin.edge_class2id

    def dump_to_file(self, fname):
        with open(fname, 'wb+') as f:
            pickle.dump(self, f)

    def add_metapath(self, metapath):
        if metapath in self.metapath_matrix:
            return
        self.metapath_matrix[metapath] = self.calculate_meta_path_adjacency_matrix(metapath)

    def relation_prediction(self, from_id, to_id, r):
        if r not in self.metapath_matrix:
            self.add_metapath(r)

        m = self.metapath_matrix[r]

        if m[from_id][to_id] > 0:
            return 1
        return 0

    def get_random_vertex(self, metapath, seed=None):
        random.seed(seed)
        from_class = metapath[0][0]
        from_id = random.sample(self.node_class[from_class], 1)

        to_class = metapath[-1][-1]
        to_id = random.sample(self.node_class[to_class], 1)

        return from_id, to_id

    def generate_next_node_pool(self):
        node_pool = {}
        for from_id in self.graph:
            node_pool[from_id] = []
            for to_id in self.graph[from_id]:
                for edge_id, weight in self.graph[from_id][to_id].items():
                    node_pool[from_id] += [(to_id, edge_id)] * int(weight * 10)
                # node_pool[from_id] += [(to_id, edge_id)]
        # print(node_pool[130618])
        self.next_node_pool = node_pool

    def generate_a_random_walk(self, from_node_id, length, keep_immediate_nodes=True):
        if from_node_id not in self.graph:
            return None
        else:
            walk = [from_node_id]
            node = from_node_id

            if not self.next_node_pool:
                self.generate_next_node_pool()

            for _ in range(length):
                if node not in self.graph:
                    return walk
                to_id, edge_id = random.choice(self.next_node_pool[node])

                walk.append(edge_id)
                if keep_immediate_nodes:
                    walk.append(to_id)
                node = to_id
            if not keep_immediate_nodes:
                walk.append(to_id)
        # print(walk)
        line = walk
        str_line = [str(x) for x in line]
        ll = '\t'.join(str_line)
        with open('./data/DBpedia/walk_10_100_test', 'a') as f:
            f.write(ll + '\n')

        return walk

    def generate_random_walks(self, num_repeat, length, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        if not self.next_node_pool:
            self.generate_next_node_pool()

        count = 0
        for i in tqdm(range(num_repeat)):
            keys = list(self.graph.keys())
            random.shuffle(keys)

            for node_id in keys:
                count += 1
                walk = self.generate_a_random_walk(node_id, length)
            # if count >9:
            # 	print(count,'end')
            # 	break
            # print(walk)
            # if len(walk) > 1:
            # 	yield walk
        print('totally ' + str(count) + ' paths generated.')

    def is_connected(self):
        visited = [False] * len(self.node_id)
        queue = [0]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue

            for i in self.graph[node]:
                if not visited[i]:
                    queue.append(i)
            visited[node] = True
        return False not in visited

    def divide_into_connected_subgraphs(self):
        result = []
        id_to_node = {v: k for k, v in self.node_id.items()}
        id_to_class = {}
        for c in self.node_class:
            for n in self.node_class[c]:
                id_to_class[n] = c
        edge_id_to_class = {v: k for k, v in self.edge_class_id.items()}
        while len(self.node_id) > 0:
            node_list = list(self.node_id.keys())
            node_name = node_list[0]
            node_id = self.node_id.pop(node_name)
            queue = [node_id]
            graph = HINGraph()
            print('graph created')
            while queue:
                from_node = queue.pop(0)
                if from_node not in self.graph:
                    continue
                for to_node in self.graph[from_node]:
                    # add all edges to the new graph
                    for edge_class_id in self.graph[from_node][to_node]:
                        graph.add_edge(id_to_node[from_node], id_to_class[from_node],
                                       id_to_node[to_node], id_to_class[to_node], edge_id_to_class[edge_class_id])
                    # add to_node to queue if it's still in self.node_id
                    if id_to_node[to_node] in self.node_id:
                        queue.append(to_node)
                        # remove it from self.node_id
                        self.node_id.pop(id_to_node[to_node])
            print('subgraph num:', len(result))
            result.append(graph)
            print('node num:', graph.node_num())
            print('edge num:', graph.edge_num())
        return result

    def random_remove_edges(self, rate=0.9, seed=None, both_sides=True):
        delete_edges = []
        random.seed(seed)
        remain_num = int(self.edge_num() * rate)

        while self.edge_count > remain_num:
            from_node = random.choice(list(self.graph.keys()))
            if len(self.graph[from_node]) == 1:
                continue
            to_node = random.choice(list(self.graph[from_node].keys()))
            if both_sides and len(self.graph[to_node]) == 1:
                continue
            edge_class = random.choice(list(self.graph[from_node][to_node].keys()))
            weight = self.graph[from_node][to_node].pop(edge_class)
            if len(self.graph[from_node][to_node]) == 0:
                self.graph[from_node].pop(to_node)
            delete_edges.append((from_node, to_node, edge_class, weight))
            self.edge_count = self.edge_count - 1

            if both_sides:
                edge_class2 = self.get_reverse_edge(edge_class)
                weight2 = self.graph[to_node][from_node].pop(edge_class2)
                if len(self.graph[to_node][from_node]) == 0:
                    self.graph[to_node].pop(from_node)
                delete_edges.append((to_node, from_node, edge_class2, weight2))
                self.edge_count = self.edge_count - 1
            if self.edge_count % 10000 == 0:
                print('remove 10000 edge success,edge_count:', self.edge_count,'remain_num:',remain_num)
        return delete_edges

    def random_remove_selected_edge(self, given_type, rate=0.1, seed=None, both_sides=False):
        deleted_edges = []
        random.seed(seed)
        given_id = self.edge_class_id[given_type]

        for from_node in list(self.graph.keys()):
            for to_node in list(self.graph[from_node].keys()):
                for edge_id in list(self.graph[from_node][to_node].keys()):
                    if edge_id == given_id and random.random() < rate:
                        # delete it
                        weight = self.graph[from_node][to_node].pop(edge_id)
                        # add it to deleted_edges
                        deleted_edges.append((from_node, to_node, edge_id, weight))
                        self.edge_count -= 1

                        if both_sides:
                            # get the reversed edge class
                            edge_class2 = self.get_reverse_edge(edge_id)
                            # weight is not used at this stage
                            weight2 = self.graph[to_node][from_node].pop(edge_class2)
                            if len(self.graph[to_node][from_node]) == 0:
                                self.graph[to_node].pop(from_node)
                            deleted_edges.append((to_node, from_node, edge_class2, weight2))
                            self.edge_count = self.edge_count - 1
                if len(self.graph[from_node][to_node]) == 0:
                    self.graph[from_node].pop(to_node)
            if len(self.graph[from_node]) == 0:
                self.graph.pop(from_node)

        return deleted_edges

    def find_diff_constant(self):
        l = sorted(self.edge_class_id.keys())
        diff = int(len(l) / 2)
        constant = l[diff] - l[0]
        base = l[diff - 1]
        return constant, base

    def get_reverse_edge(self, given_id):
        if not self.reverse_list:
            self.reverse_list = {}
            if type(next(iter(self.edge_class_id.keys()))) is int:
                # find the diff constant
                constant, base = self.find_diff_constant()
                for cls, idx in self.edge_class_id.items():
                    if cls <= base:
                        r_cls = cls + constant
                    else:
                        r_cls = cls - constant
                    self.reverse_list[idx] = self.edge_class_id[r_cls]
            else:
                for cls, idx in self.edge_class_id.items():
                    r_cls = cls[::-1]
                    self.reverse_list[idx] = self.edge_class_id[r_cls]
        return self.reverse_list[given_id]

    def remove_given_node_class(self, t):
        for from_node in self.node_class[t]:
            if from_node not in self.graph:
                continue
            for to_node in self.graph[from_node]:
                tmp = self.graph[to_node].pop(from_node)
                self.edge_count -= len(tmp) * 2
                if len(self.graph[to_node]) == 0:
                    self.graph.pop(to_node)
            tmp = self.graph.pop(from_node)

        self.node_id = {k: v for k, v in self.node_id.items() if v not in self.node_class[t]}
        self.node_class.pop(t)

    def sample_subgraph(self, keep_rate=0.1, seed=None):
        random.seed(seed)

        id_to_node = {v: k for k, v in self.node_id.items()}
        id_to_class = {}
        for c in self.node_class:
            for n in self.node_class[c]:
                id_to_class[n] = c
        edge_id_to_class = {v: k for k, v in self.edge_class_id.items()}
        sampled_nodes = set()
        for node_id, _ in id_to_class.items():
            if random.random() < keep_rate:
                sampled_nodes.add(node_id)

        graph = HINGraph()
        for from_node in sampled_nodes:
            for to_node in self.graph[from_node]:
                if to_node in sampled_nodes:
                    for edge_class_id in list(self.graph[from_node][to_node].keys()):
                        graph.add_edge(id_to_node[from_node], id_to_class[from_node], id_to_node[to_node],
                                       id_to_class[to_node], edge_id_to_class[edge_class_id])
        self.get_node_class(graph)
        return graph

    def subgraph(self, visited_nodes):
        id_to_node = {v: k for k, v in self.node_id.items()}
        id_to_class = {}
        for c in self.node_class:
            for n in self.node_class[c]:
                if n in visited_nodes:
                    id_to_class[n] = c
        edge_id_to_class = {v: k for k, v in self.edge_class_id.items()}

        graph = HINGraph()

        sorted_dict = sorted(id_to_class.items(), key=lambda d: d[1])
        for n, c in sorted_dict:
            graph.node_id[id_to_node[n]] = len(graph.node_id)
        # load the edges
        for from_node in visited_nodes:
            for to_node in self.graph[from_node]:
                if to_node in visited_nodes:
                    for edge_class_id in list(self.graph[from_node][to_node].keys()):
                        graph.add_edge(id_to_node[from_node], id_to_class[from_node], id_to_node[to_node],
                                       id_to_class[to_node], edge_id_to_class[edge_class_id])
        print('num of edges', graph.edge_num())
        return graph

    def get_k_hop_neighbors(self, k=6, min=100000, max=1000000, node_id=None):
        if not node_id:
            node_id = random.sample(self.graph.keys(), 1)[0]

        visited_nodes, pre_hop, cur_hop = {node_id}, {node_id}, set()
        # for each hop
        for i in range(1, k + 1):
            # for each new node in that hop
            for from_node in pre_hop:
                # for each of its neighbor
                for to_node in self.graph[from_node]:
                    if to_node not in visited_nodes:
                        visited_nodes.add(to_node)
                        cur_hop.add(to_node)
            c = len(visited_nodes)
            if c >= min:
                if c <= max:
                    print('node_id', node_id)
                    print('num of nodes', c)
                    return visited_nodes
                else:
                    return None
            pre_hop = cur_hop
            cur_hop = set()

    def get_connected_nodes(self, heads, r_id):
        results = set()
        for head in heads:
            if head not in self.graph.keys() or len(self.graph[head].keys()) == 0:
                continue
            for tail in self.graph[head].keys():
                if r_id in self.graph[head][tail].keys():
                    results.add(tail)
        return results

    def get_dest(self, head, relation_list):
        tmp = {head}
        for r_id in relation_list:
            tmp = self.get_connected_nodes(tmp, r_id)
            if len(tmp) == 0:
                return None
        return tmp

    def is_metapath_between_pairs(self, datain):
        from_node_id = datain[0]
        to_node_id = datain[-1]
        relation_list = datain[1:-1]

        vaild_to_node_list = self.get_dest(from_node_id, relation_list)
        if not vaild_to_node_list:
            return False
        if to_node_id in vaild_to_node_list:
            return True
        return False

    def output_triplets(self, filename):
        if os.path.isfile(filename):
            return None
        else:
            with open(filename, 'w+') as f:
                for from_node in list(self.graph.keys()):
                    for to_node in list(self.graph[from_node].keys()):
                        for edge_id in list(self.graph[from_node][to_node].keys()):
                            f.write(str(from_node) + "\t" + str(edge_id) + "\t" + str(to_node) + "\n")







