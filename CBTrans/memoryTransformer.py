import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from repblobk import LocalPerceptron
from relative_embedding import GridRelationalEmbedding
from encoders import MultiLevelEncoder
from decoders import TransformerDecoderLayer

class MLP(nn.Module):
    def __init__(self, d_model,dff1, rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, dff1)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(dff1, d_model)
        self.dropout = nn.Dropout(rate)

    def forward(self, X):
        X = self.fc1(X)
        X = self.GELU(X)
        X = self.fc2(X)
        X = self.dropout(X)
        return X
class MemoryTransformer(nn.Module):
    def __init__(self, N_enc, N_dec,  vocab_size,d_model,num_classes):
        #num_layers1,num_layers, d_model, num_heads, dff, num_classes, dff1,vocab_size, dropout=dropout
        super(MemoryTransformer, self).__init__()

        self.encoder= MultiLevelEncoder(N_enc,d_model = 1024, d_k = 128, d_v = 128, h = 8, d_ff = 2048, dropout = .1,
                                                                 identity_map_reordering = False, attention_module = None, attention_module_kwargs = None)



        self.decoder = TransformerDecoderLayer(vocab_size,N_dec,max_len=50,d_model = 1024, d_k = 128, d_v = 128, h = 8, d_ff = 2048, dropout = .1,
                                                                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None)
        self.mlp = MLP(d_model=1024, dff1=4096)
        self.vocab_size = vocab_size
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        decode_lengths = [c - 1 for c in caption_lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        enc_output, mask_enc = self.encoder(encoder_out)
        encoder_out1 = enc_output.clone()
        dec_output,encoder_out1 = self.decoder(encoded_captions, enc_output, encoder_out1,mask_enc)

        pooled_features = F.avg_pool1d(encoder_out1.permute(0, 2, 1), kernel_size=encoder_out1.size(1))#.squeeze(2)
        #print(pooled_features.shape)                                                       ````

        pooled_features = torch.flatten(pooled_features,1)
        mlp_output = self.mlp(pooled_features)
        claout = mlp_output + pooled_features
        #claout = pooled_features
        #print(pooled_features.shape)'''
        logits = self.classifier(claout)
        logits = F.log_softmax(logits, dim=-1)
        #print(dec_output.shape)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            preds = self.final_layer(self.dropout(dec_output[:batch_size_t, t, :]))
            # preds = self.final_layer(self.dropout(embeddings[:, t, :]))
            # print("preds.shape: ", preds.shape)
            # print(len(predictions[:batch_size_t, t, :]))
            predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha'''
        #print("ll",predictions.shape)
        predictions = F.log_softmax(predictions, dim=-1)
        #print("kk",predictions.shape)
        return predictions, encoded_captions, decode_lengths,logits


