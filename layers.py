import torch
import torch.nn as nn
import torch.nn.functional as F

class My_Bilinear(torch.nn.Module):

    def __init__(self,x):
        super(My_Bilinear, self).__init__()
        self.linear=torch.nn.Linear(x,x)

    def forward(self,A,B):
        tmp=self.linear(A)
        out=torch.mm(tmp,B.T)
        #out=torch.sigmoid(out)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = My_Bilinear(n_h)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):

        c=torch.unsqueeze(c,0)

        h_pl=torch.squeeze(h_pl)
        h_mi=torch.squeeze(h_mi)

        sc_1 = torch.squeeze(self.f_k(c,h_pl))
        sc_2 = torch.squeeze(self.f_k(c,h_mi))

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2))

        return logits




class AvgReadout(torch.nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            #return torch.mean(seq, 1)
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)


class SemanticAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(SemanticAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.q = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_uniform_(self.q.data, gain=1.414)
        self.Tanh = nn.Tanh()

    # input (PN)*F
    def forward(self, input, P):
        h = torch.mm(input, self.W)
        # h=(PN)*F'
        h_prime = self.Tanh(h + self.b.repeat(h.size()[0], 1))
        # h_prime=(PN)*F'
        semantic_attentions = torch.mm(h_prime, torch.t(self.q)).view(P, -1)
        # semantic_attentions = P*N
        N = semantic_attentions.size()[1]
        semantic_attentions = semantic_attentions.mean(dim=1, keepdim=True)
        # semantic_attentions = P*1
        semantic_attentions = F.softmax(semantic_attentions, dim=0)

        semantic_attentions = semantic_attentions.view(P, 1, 1)
        semantic_attentions = semantic_attentions.repeat(1, N, self.in_features)

        input_embedding = input.view(P, N, self.in_features)

        h_embedding = torch.mul(input_embedding, semantic_attentions)
        h_embedding = torch.sum(h_embedding, dim=0).squeeze()

        return h_embedding


