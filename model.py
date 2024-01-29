import torch
from torch import nn
import time
import torch.nn.functional as F
from self_Att import SelfAttModel

class Able(nn.Module):
    def __init__(self, model_params):
        super(Able, self).__init__()
        self.mp_encoder = model_params['mp_encoder']
        self.node_att = model_params['node_att']
        self.loss_switch = 'cross1'
        self.mode_switch = model_params['mod_switch']
        self.self_att = SelfAttModel(model_params["NODE_EMBEDDING_DIM"]*2,1)
        self.sel_att_linear = nn.Linear(model_params['NODE_EMBEDDING_DIM']*2, model_params['NODE_EMBEDDING_DIM']*2)
        self.embedding_all_node = nn.Embedding(model_params['NODE_NUM'], model_params["NODE_EMBEDDING_DIM"])

        # head node embedding
        self.embedding_head = nn.Embedding(model_params['NODE_NUM'],model_params["NODE_EMBEDDING_DIM"])
        self.embedding_tail = nn.Embedding(model_params['NODE_NUM'],model_params['NODE_EMBEDDING_DIM'])
        self.embedding_node_type = nn.Embedding(model_params['NODE_TYPE_NUM'], model_params['NODE_TYPE_EMBEDDING_DIM'])
        self.embedding_edge_type = nn.Embedding(model_params['EDGE_NUM'],model_params['EDGE_EMBEDDING_DIM'])
        nn.init.xavier_uniform_(self.embedding_head.weight)
        nn.init.xavier_uniform_(self.embedding_node_type.weight)
        nn.init.xavier_uniform_(self.embedding_tail.weight)
        nn.init.xavier_uniform_(self.embedding_edge_type.weight)
        nn.init.xavier_uniform_(self.embedding_all_node.weight)
        # find node's type
        self.look_node_type = model_params["LOOKUP_TABLE"]
        # attention
        self.bilistm = nn.LSTM(input_size=model_params['EDGE_EMBEDDING_DIM'],hidden_size=model_params['LSTM_HIDDEN_SIZE'],bidirectional=True,
                               num_layers=1,batch_first=True,dropout=0)
        # attention node
        self.bilistm_node = nn.LSTM(input_size=model_params['EDGE_EMBEDDING_DIM'],hidden_size=model_params['LSTM_HIDDEN_SIZE'],bidirectional=True,
                               num_layers=1,batch_first=True,dropout=0)
        self.liner_node = nn.Linear(model_params['LSTM_HIDDEN_SIZE'] * 2, model_params['EDGE_EMBEDDING_DIM'])
        # reduce the output of bilstm
        self.liner = nn.Linear(model_params['LSTM_HIDDEN_SIZE']*2,model_params['EDGE_EMBEDDING_DIM'])
        #     attention matrix 1tou
        self.a = nn.Parameter(torch.randn(1, model_params['ATTENTION_HEAD'],model_params['EDGE_EMBEDDING_DIM']))
        self.a1 = nn.Parameter(torch.randn(1, model_params['EDGE_EMBEDDING_DIM']))
        self.attn1 = nn.Linear(model_params['EDGE_EMBEDDING_DIM'], model_params['ATTENTION_HEAD'], bias=False)
        self.attn1_node = nn.Linear(model_params['EDGE_EMBEDDING_DIM'], model_params['ATTENTION_HEAD'], bias=False)
        nn.init.xavier_uniform_(self.attn1.weight)
        nn.init.xavier_uniform_(self.attn1_node.weight)
        nn.init.xavier_uniform_(self.liner.weight)

        self.liner1 = nn.Linear(model_params['EDGE_EMBEDDING_DIM'],model_params['FC_HIDDEN_SIZE'])
        self.liner2 = nn.Linear(model_params['FC_HIDDEN_SIZE'],1)
        self.liner_cross = nn.Linear(model_params['FC_HIDDEN_SIZE'], 2)
        nn.init.xavier_uniform_(self.liner2.weight)
        nn.init.xavier_uniform_(self.liner1.weight)
        nn.init.xavier_uniform_(self.liner_cross.weight)


        #  activate functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attSoftmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.norm_node = nn.LayerNorm(model_params['EDGE_EMBEDDING_DIM'])


    def forward(self, x):
        if self.mode_switch == 1:
            pos, neg = torch.chunk(x, chunks=2, dim=0)
            # print('pos:', pos)
            pos_ = self.calculate(pos)
            neg_ = self.calculate(neg)
            # print('pos_:', pos_)
            # print('neg_:', neg_)
            # 交叉熵损失函数，二分类
            los = neg_ - pos_ + 0.4
            los = self.relu(los)
            los = torch.sum(los)
            # print(los)
            return los
        elif self.mode_switch == 2:
            return self.calculate(x)

    def calculate(self, x):
        if self.node_att == "self_att":
            # print('x',x)
            node_id = x[:, 0::2]
            # print(node_id.shape)
            # reshape node id to get node type id
            node_type_id_index = node_id.reshape(-1)
            node_type_id = torch.index_select(self.look_node_type, 0, node_type_id_index)
            # print(node_type_id)
            node_type_id = node_type_id.view(node_id.shape)
            # print(node_type_id)
            node_embedding = self.embedding_all_node(node_id)
            node_type_embedding = self.embedding_node_type(node_type_id)
            # print(node_embedding)
            # print(node_type_embedding)
            # node em cat node type em
            ah = torch.cat((node_embedding, node_type_embedding), dim=-1)
            # print(ah)
            # self attention caculate node posisition and weight
            ah = self.self_att(ah)
            # print(ah)
            # 平均
            ah = torch.mean(ah,dim=1)
            # print(ah)
            # 平均后再线性
            if self.mp_encoder == "linear":
                ah = self.sel_att_linear(ah)
            # print(ah)
            # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # bt = torch.ones(ah.shape)
            # bt = bt.to(device)
        elif self.node_att == "att":
            # 考虑中间节点，但是只考虑不同节点对mp嵌入的不同权重，所有节点嵌入*权重
            node_id = x[:, 0::2]
            # print(node_id.shape)
            # reshape node id to get node type id
            node_type_id_index = node_id.reshape(-1)
            node_type_id = torch.index_select(self.look_node_type, 0, node_type_id_index)
            # print(node_type_id)
            node_type_id = node_type_id.view(node_id.shape)
            # print(node_type_id)
            node_embedding = self.embedding_all_node(node_id)
            node_type_embedding = self.embedding_node_type(node_type_id)
            # print(node_embedding)
            # print(node_type_embedding)
            # node em cat node type em
            ah = torch.cat((node_embedding, node_type_embedding), dim=-1)
            # print('ah',ah)
            # 多头注意力，得到多个权重，将所有权重*原矩阵 再相加，最后leck relu
            att = self.attn1_node(ah)
            # print('att',att)
            att1 = torch.permute(att, [2, 0, 1])
            # print('att1',att1)
            weights = self.attSoftmax(att1)
            # print('att1 weight',weights)
            weights = torch.unsqueeze(weights, dim=-1)
            # print('weig',weights)
            # 广播机制，得到heads 个加权后的结果。全部相加
            cp = ah * weights
            cp = torch.sum(cp, dim=0)
            # 节点的嵌入全部加起来，attention的结果相加
            cp = torch.sum(cp, dim=1)
            ah = self.leaky_relu(cp)
            ah = self.norm_node(ah)
        elif self.node_att == "att_bi":
            # 考虑中间节点，同时考虑节点的顺序信息
            node_id = x[:, 0::2]
            # print(node_id.shape)
            # reshape node id to get node type id
            node_type_id_index = node_id.reshape(-1)
            node_type_id = torch.index_select(self.look_node_type, 0, node_type_id_index)
            # print(node_type_id)
            node_type_id = node_type_id.view(node_id.shape)
            # print(node_type_id)
            node_embedding = self.embedding_all_node(node_id)
            node_type_embedding = self.embedding_node_type(node_type_id)
            # print(node_embedding)
            # print(node_type_embedding)
            # node em cat node type em
            ah = torch.cat((node_embedding, node_type_embedding), dim=-1)

            # 节点的顺序信息获取，然后加上ah，最后再进入attention
            hr, (h0, h1) = self.bilistm_node(ah)
            # print('hr:', hr)

            # 降维,此处使用的hiden size和边的一致
            pn = self.liner_node(hr)
            # print(pn.shape)
            pn = self.norm_node(pn)
            pn = self.sigmoid(pn)
            # 节点嵌入+节点位置信息
            ah = ah + pn
            # print('ah',ah)
            # 多头注意力，得到多个权重，将所有权重*原矩阵 再相加，最后leck relu
            att = self.attn1_node(ah)
            # print('att',att)
            att1 = torch.permute(att, [2, 0, 1])
            # print('att1',att1)
            weights = self.attSoftmax(att1)
            # print('att1 weight',weights)
            weights = torch.unsqueeze(weights, dim=-1)
            # print('weig',weights)
            # 广播机制，得到heads 个加权后的结果。全部相加
            cp = ah * weights
            cp = torch.sum(cp, dim=0)
            # 节点的嵌入全部加起来，attention的结果相加
            cp = torch.sum(cp, dim=1)
            ah = self.relu(cp)
        elif self.node_att == "nonode":
            pass
        else:
            h_node_id = x[:, :1]
            # print(h_node_id)
            h_node_id = torch.squeeze(h_node_id, dim=1)
            # print('h:', h_node_id)

            # get node's type id
            h_type_id = torch.index_select(self.look_node_type, 0, h_node_id)
            h_type_id = torch.squeeze(h_type_id, dim=1)
            # print('h_type_id:', h_type_id)

            # get node and node type embeddings
            h = self.embedding_head(h_node_id)
            h_type = self.embedding_node_type(h_type_id)
            # Concatenation node embedding
            ah = torch.cat((h, h_type), dim=1)


            t_node_id = x[:, -1:]
            t_node_id = torch.squeeze(t_node_id, dim=1)
            # print('t:', t_node_id)
            t_type_id = torch.index_select(self.look_node_type, 0, t_node_id)
            t_type_id = torch.squeeze(t_type_id, dim=1)
            # print('t_type_id:', t_type_id)
            t = self.embedding_tail(t_node_id)
            t_type = self.embedding_node_type(t_type_id)
            # print('h_node_embedding:', h)
            # print('h_type_embedding:', h_type)
            # Concatenation node embedding
            bt = torch.cat((t,t_type), dim=1)
            ah = ah * bt


        # attention-base meta-path embedding
        edge_type_ids = x[:, 1:-1]
        edge_type_ids = edge_type_ids[::, ::2]
        # edge_type_ids = torch.squeeze(edge_type_ids, dim=1)
        # print('edge_type_ids',edge_type_ids)
        #  (batch,seq_length,input_size)
        zr = self.embedding_edge_type(edge_type_ids)
        # print('zr:', zr)
        hr, (h0, h1) = self.bilistm(zr)
        # print('hr:', hr)

        # 降维
        pr = self.liner(hr)
        pr = self.sigmoid(pr)
        # print('pr:', pr)
        # zrr =>zr +pr 融入位置信息（顺序信息）
        zrr = zr + pr
        # print('zrr:', zrr)

        # attention1 单头
        # aT = self.a1
        # print('aT', aT)
        # att = aT * zrr
        # att = torch.sum(att, dim=-1)
        # att = torch.unsqueeze(att, dim=-1)
        # # print('att:',att)
        # # print('att1',torch.dot(zrr, aT))
        # weights = self.softmax(self.leaky_relu(att))
        # print('weights:', weights)

        # 多头注意力，得到多个权重，将所有权重*原矩阵 再相加，最后leck relu
        att = self.attn1(zrr)
        # print('att',att)
        att1 = torch.permute(att, [2,0,1])
        # print('att1',att1)
        weights = self.attSoftmax(att1)
        # print('att1 weight',weights)
        weights = torch.unsqueeze(weights, dim=-1)
        # print('weig',weights)
        # 广播机制，得到heads 个加权后的结果。全部相加
        cp = zrr * weights
        # print('cp0:',cp)

        cp = torch.sum(cp, dim=0)
        # print('cp',cp)

        # 边的嵌入全部加起来，attention的结果相加
        cp = torch.sum(cp,dim=1)
        cp = self.leaky_relu(cp)
        # print('cp:', cp)

        # trip prediction
        # f(h,p,t)
        # print(ah)
        # print(bt)

        # print(ah * bt)
        # if self.node_att in ['att','att_bi','self_att']:
        # self.sigmoid(cp)
        trip = self.sigmoid(cp)
        # trip = ah * cp
        # else:
        #     trip = (ah * bt) * self.sigmoid(cp)
        trip = self.norm_node(trip)
        # print('trip:',trip.shape)
        pr = self.liner1(trip)
        # print('pr:', pr.shape)
        # pr : （0,1）

        if self.loss_switch == 'cross':
            pr = self.liner_cross(self.relu(pr))
            pr = self.softmax(pr)
            # print(pr)
        else:
            pr = self.liner2(self.relu(pr))
            pr = self.sigmoid(pr)
        if self.mode_switch == 2:
            # 预测评分
            pr = torch.squeeze(pr, dim=-1)
            pr = 4 * pr + 1

            # print('rate is?',pr)
        # print('pr:', pr)
        # time.sleep(111111)
        return pr


if __name__ =='__main__':
    # model = Able(model_params)

    # print(list(model.named_parameters()))
    # for k,v in model.named_parameters():
    #     print(k)

    # em=nn.Embedding(5, 3)
    # print(em.weight)
    # index=torch.tensor([1,2])
    # a =torch.tensor(([3,4]))
    # print(index)
    # print(em(index))

    # loss = nn.MSELoss(reduction='sum')
    # input = torch.randn(2, 1, requires_grad=True)
    # print(input)
    # target = torch.randn(2, 1)
    # print(target)
    # output = loss(input, target)
    # print(output)
    # output.backward()

    x = torch.randn(1,1,4)
    # print(x)
    x=torch.transpose(x,1,2)
    print(x)
    y=torch.randn(2,2,4)
    print('y:',y)
    # z=x*y
    z=torch.matmul(y,x)
    print('z:',z)
    print(z.sum(-1))
    # aT = torch.transpose(x, 1, 2)

    # print(aT)

    #  heads, out_channels
    # a=torch.Tensor(1,2,3)
    # print(a)
    # b=a.view(-1,3,2)
    # print(b)
    #
    # indices = torch.tensor([1, 2])
    # a = torch.index_select(x, 0, indices)
    # print(a)

    # x=torch.tensor([[19, 1754, 11, 1755, 92],
    #         [50, 1756, 51, 1799, 90]])
    # print(x)
    # print(x[::,::2])
