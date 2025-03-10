import numpy as np
import torch
from torch import nn
import math

class ScaledDotProductAttentionMemory(nn.Module):
    '''
    Scaled dot-product attention with memory
    '''

    def __init__(self, d_model, d_k, d_v, h, m):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param m: Number of memory slots
        '''
        super(ScaledDotProductAttentionMemory, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.m_k = nn.Parameter(torch.FloatTensor(1, m, h * d_k))
        self.m_v = nn.Parameter(torch.FloatTensor(1, m, h * d_v))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.m = m

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        #nn.init.normal_(self.m_k, 0, 1 / self.d_k)
        #nn.init.normal_(self.m_v, 0, 1 / self.m)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, relative_pos=None,attention_mask=None, attention_weights=None):
        #queries, keys, values,relative_pos, attention_mask, attention_weights
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        m_k = np.sqrt(self.d_k) * self.m_k.expand(b_s, self.m, self.h * self.d_k)
        m_v = np.sqrt(self.m) * self.m_v.expand(b_s, self.m, self.h * self.d_v)

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = torch.cat([self.fc_k(keys), m_k], 1).view(b_s, nk + self.m, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = torch.cat([self.fc_v(values), m_v], 1).view(b_s, nk + self.m, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = torch.cat([att[:, :, :, :nk] * attention_weights, att[:, :, :, nk:]], -1)
        if attention_mask is not None:
            att[:, :, :, :nk] = att[:, :, :, :nk].masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out
class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

def PositionalEncoding(x,d_model,len_q):


    pe = torch.zeros(len_q, d_model)
    position = torch.arange(0, len_q, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    out = pe.unsqueeze(0)[:, :len_q, :].to(x.device)

    return out

def softmax_with_bias(x,bias):
    # x = x.cuda(1).data.cpu().numpy()
    #x = torch.tensor(x).cuda(1).data.cpu().numpy()
    x = x * bias
    exp =np.exp(x)
    return exp / np.sum(exp,axis=-1,keepdims=True)


def softmax_with_bias1(x,bias):
    # x = x.cuda(1).data.cpu().numpy()
    #x = torch.tensor(x).cuda(1).data.cpu().numpy()
    x = x * bias
    exp =torch.exp(x)
    return exp /torch.sum(exp,dim=-1,keepdim=True)
class MultiHeadAttention2(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention2, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttentionMemory(d_model=d_model, d_k=d_k, d_v=d_v, h=h, m=30)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)



    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):
        out = self.attention(queries, keys, values,relative_pos, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out

class MemoryMultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None,num_memory=20):
        super(MemoryMultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.k_memory = nn.Parameter(torch.randn(1, num_memory, d_model) * (1 / np.sqrt(d_k)))
        self.v_memory = nn.Parameter(torch.randn(1, num_memory, d_model) * (1 / np.sqrt(num_memory)))
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)


    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):
        #print(keys.shape)
        k_memory = self.k_w(self.k_memory)
        v_memory = self.v_w(self.v_memory)
        keys = torch.cat([keys, k_memory.expand(keys.size(0), -1, -1)], dim=1)
        values = torch.cat([values, v_memory.expand(values.size(0), -1, -1)], dim=1)
        #print(keys.shape)
        out = self.attention(queries, keys, values,relative_pos, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out

class ScaledDotProductAttention1(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self,d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention1, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment
        self.ratio = nn.Parameter(torch.tensor((0.8)))

        self.bias = nn.Parameter(torch.tensor(0.7))

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        #print(b_s,nq,nk)
        #print(queries.shape)
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # print(q.shape)
        # print(q.shape)
        pos = PositionalEncoding(q, self.d_k, nq)
        # pos = self.pe(q,nq)
        # print(pos.shape)
        # pos1 = self.pe(k,nk)
        pos1 = PositionalEncoding(k.transpose(-1, -2), self.d_k, nk)
        # print(pos1.shape)
        pos2 = torch.matmul(pos, pos1.transpose(-1, -2))
        # print("A",pos2.shape)
        # print("B",att.shape)
        att = att + pos2  # HAPE操作结束
        #print(att.shape)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        if relative_pos is not None:
            wg = torch.log(torch.clamp(relative_pos, min=1e-6))
            att = att + wg

        ratio = torch.sigmoid(self.ratio)
        top_k = int(ratio*att.shape[-1])
        val,indices = torch.topk(att,top_k,dim=-1)
        #print(val.shape)
        filter_value = -float('inf')
        index = att<val[:,:,:,-1].unsqueeze(-1).repeat(1,1,1,att.shape[-1])
        att_ = att.detach()
        att_[index] = filter_value
        #b = self.softmax(scores_)
        # print(a[:,:,:,1])
        #pos =
        b = softmax_with_bias1(att_,self.bias)
        # #print(a[:,:,:,1])
        #
        #b = torch.tensor(b)
        b = b.clone().detach()
        att = self.dropout(b)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out





class MultiHeadAttention1(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention1, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention1(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)



    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):
        out = self.attention(queries, keys, values,relative_pos, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out

class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)



    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):
        out = self.attention(queries, keys, values,relative_pos, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out

class MemoryMultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None,num_memory=10):
        super(MemoryMultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.k_memory = nn.Parameter(torch.randn(1, num_memory, d_model) * (1 / np.sqrt(d_k)))
        self.v_memory = nn.Parameter(torch.randn(1, num_memory, d_model) * (1 / np.sqrt(num_memory)))
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)


    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None):

        #print(keys.shape)
        k_memory = self.k_w(self.k_memory)
        v_memory = self.v_w(self.v_memory)
        keys = torch.cat([keys, k_memory.expand(keys.size(0), -1, -1)], dim=1)
        values = torch.cat([values, v_memory.expand(values.size(0), -1, -1)], dim=1)


        out = self.attention(queries, keys, values,relative_pos, attention_mask, attention_weights)
        out = self.dropout(out)
        out = self.layer_norm(queries + out)
        return out