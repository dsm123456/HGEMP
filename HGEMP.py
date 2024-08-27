import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import Linear
from layers import Discriminator
from layers import AvgReadout
from layers import SemanticAttentionLayer
import torch.nn.functional as F
import numpy as np


class MPLHGE(torch.nn.Module):
    def __init__(self,input_dim,att_dim,dropout_rate=0.):
        super(MPLHGE, self).__init__()
        #super parameters:
        self.dropout_rate = dropout_rate
        self.linear=Linear(-1,input_dim)

        self.semantic_level_attention=SemanticAttentionLayer(input_dim, att_dim)
        self.readout = AvgReadout()
        self.disc = Discriminator(input_dim)
        self.sig = torch.sigmoid

        self.conv_1 = GCNConv(input_dim, input_dim)
        self.conv_2 = GCNConv(input_dim, input_dim)


    def forward(self,graph,summary_vetor,nodeType,Batch_size):

        x_dict=graph.x_dict
        edge_index_dict=graph.edge_index_dict

        X_origin=x_dict[nodeType]

        idx = np.random.permutation(graph[nodeType].num_nodes)
        X_origin_shuffle = X_origin[idx, :]


        X_origin=self.linear(X_origin) #输入是one-hot时可以参考使用torch.nn.embedding()，减少模型输入规模
        X_origin=F.relu(X_origin)
        X_origin_shuffle=self.linear(X_origin_shuffle)
        X_origin_shuffle=F.relu(X_origin_shuffle)






        GNN_conv_list_1=[]
        GNN_conv_list_2=[]
        for edge in edge_index_dict.values():
            X = self.conv_1(X_origin, edge)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout_rate, training=self.training)

            X = self.conv_2(X, edge)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout_rate, training=self.training)


            X_shuffle = self.conv_1(X_origin_shuffle, edge)
            X_shuffle = F.relu(X_shuffle)
            X_shuffle = F.dropout(X_shuffle, p=self.dropout_rate, training=self.training)

            X_shuffle = self.conv_2(X_shuffle, edge)
            X_shuffle = F.relu(X_shuffle)
            X_shuffle = F.dropout(X_shuffle, p=self.dropout_rate, training=self.training)


            ###########
            GNN_conv_list_1.append(X)
            GNN_conv_list_2.append(X_shuffle)




        muilt_gcn_out=torch.cat(GNN_conv_list_1,dim=0)
        muilt_gcn_out_nagtive = torch.cat(GNN_conv_list_2,dim=0)


        Att_out = self.semantic_level_attention(muilt_gcn_out , len(edge_index_dict))
        Att_out_nagtive = self.semantic_level_attention(muilt_gcn_out_nagtive, len(edge_index_dict))


        Att_out=Att_out[:Batch_size]
        Att_out_nagtive=Att_out_nagtive[:Batch_size]

        summary_vetor_in_model = self.readout(Att_out, None)

        summary_vetor = torch.stack((summary_vetor, summary_vetor_in_model), dim=0)
        summary_vetor = torch.mean(summary_vetor,dim=0)
        #summary_vetor = self.sig(summary_vetor)

        samp_bias1 = None
        samp_bias2 = None

        ret = self.disc(summary_vetor, Att_out, Att_out_nagtive, samp_bias1, samp_bias2)
        return ret, summary_vetor


    def get_embedding(self,graph,nodeType,target_id_set=None):

        x_dict = graph.x_dict
        edge_index_dict = graph.edge_index_dict

        X_origin = x_dict[nodeType]


        X_origin = self.linear(X_origin)
        X_origin = F.relu(X_origin)


        GNN_conv_list_1 = []
        for edge in edge_index_dict.values():
            X = self.conv_1(X_origin, edge)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout_rate, training=self.training)

            X = self.conv_2(X, edge)
            X = F.relu(X)
            X = F.dropout(X, p=self.dropout_rate, training=self.training)


            GNN_conv_list_1.append(X)


        muilt_gcn_out = torch.cat(GNN_conv_list_1, dim=0)
        Att_out = self.semantic_level_attention(muilt_gcn_out, len(edge_index_dict))

        embedding=Att_out
        embedding =embedding.detach().cpu()

        if target_id_set == None:
            return embedding
        else:
            return embedding[target_id_set,:]






