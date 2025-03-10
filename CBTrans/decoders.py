import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from attention import MultiHeadAttention,MultiHeadAttention2,MemoryMultiHeadAttention
from utils import sinusoid_encoding_table, PositionWiseFeedForward



class DecoderLayer(nn.Module):
    def __init__(self, d_model=1024, d_k=128, d_v=128, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MemoryMultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.enc_att1 = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1=nn.Dropout(dropout)
        self.lnorm1=nn.LayerNorm(d_model)

        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)
        self.lnorm2=nn.LayerNorm(d_model)
        self.lnorm3=nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.pwff1 = PositionWiseFeedForward(d_model, d_ff, dropout)


    def forward(self, input, enc_output, enc_out1, mask_enc_att):
        #MHA+AddNorm
        self_att = self.self_att(input, input, input, None)
        self_att = self.lnorm1(input + self.dropout1(self_att))

        # MHA+AddNorm
        enc_att = self.enc_att(self_att, enc_output, enc_output, None, mask_enc_att)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        # FFN+AddNorm
        ff = self.pwff(enc_att)

        # MHA+AddNorm
        enc_att1 = self.enc_att1(enc_out1, ff, ff, None, None)
        enc_att1 = self.lnorm3(enc_out1 + self.dropout3(enc_att1))
        # FFN+AddNorm
        ff1 = self.pwff1(enc_att1)


        return ff,ff1



def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(1000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)
class TransformerDecoderLayer(nn.Module):
    def __init__(self, vocab_size,N_dec, max_len,  d_model=1024, d_k=128, d_v=128, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):

        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

        self.N = N_dec

        self.pos_encoding = positional_encoding(1000, d_model)

        self.dropout = nn.Dropout(dropout)
    def forward(self, input, encoder_output, encoder_out1,mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        embeddings = self.word_emb(input)
        embeddings *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        embeddings += self.pos_encoding[:, :seq_len, :].to(embeddings.device)
        out = self.dropout(embeddings)

        for i, l in enumerate(self.layers):
            out,encoder_out1 = l(out, encoder_output, encoder_out1,mask_encoder)


        return out,encoder_out1
