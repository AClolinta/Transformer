import torch
import torch.nn as nn
import math
import numpy as np
import torch.optim as optim

if __name__ == '__main__':
    # S : Symbol that shows starting of decoding input
    # E : Symbol that shows starting of decoding output
    # P : Symbol that will fill in blank sequence if current batch data size is short than time steps
    sentences = [
        # enc_input           dec_input         dec_output
        ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
        ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
    ]

    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
    src_vocab_size = len(src_vocab)
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, '.': 6, 'S': 7, 'E': 8}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # enc_input max sequence length
    tgt_len = 6  # dec_input(=dec_output) max sequence length

    ################### Hyper Parameters ###################

    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention


################## Transformer ##################

def get_attn_pad_mask(self, seq_q, seq_k):
    # seq_q: [batch_size x seq_length],
    # seq_k: [batch_size x seq_length]
    # pad_attn_mask: [batch_size x seq_length x seq_length]
    # seq_len could be source_length or target_length
    # seq_len in seq_q and seq_len in seq_k maybe not equal

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size x 1 x len_k] False is 0, True is 1
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size x len_q x len_k]


def get_attn_subsequence_mask(self, seq):
    # sqe: [batch_size x seq_length]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsquence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
    subsquence_mask = torch.from_numpy(subsquence_mask).byte()
    return subsquence_mask  # [batch_size x seq_length x seq_length]


class ScaleDotProductAttention(nn.Module):
    def __int__(self):
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size x n_heads x len_q x d_k]
        # K: [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_v(=len_k) x d_v]
        # attn_mask: [batch_size x n_heads x seq_length x seq_length]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores: [batch_size x n_heads x len_q x len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size x n_heads x len_q x len_k]
        context = torch.matmul(attn, V)  # [batch_size x n_heads x len_q x d_v]

        return context, attn


class MultiHeadAttention(nn.Module):
    # Multi-head Attention module
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_q, input_k, input_v, attn_mask):
        # input_q: [batch_size x len_q x d_model]
        # input_k: [batch_size x len_k x d_model]
        # input_v: [batch_size x len_v(=len_k) x d_model]
        # attn_mask: [batch_size x seq_length x seq_length]
        residual, batch_size = input_q, input_q.size(0)

        Q = self.W_Q(input_q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size x n_heads x len_q x d_k]
        K = self.W_K(input_k).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size x n_heads x len_k x d_k]
        V = self.W_V(input_v).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # [batch_size x n_heads x len_v(=len_k) x d_v]

        # Masking to avoid the attention of the future
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size x n_heads x seq_length x seq_length]

        # Contextual vector
        context, attn = ScaleDotProductAttention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)  # [batch_size x len_q x n_heads * d_v]
        output = self.fc(context)  # [batch_size x len_q x d_model]

        return nn.LayerNorm(d_model).cuda()(output + residual), attn  # [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __int__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        # Add and Norm

        residual = inputs  # inputs: [batch_size, seq_length, d_model]
        output = self.fc(inputs)  # output: [batch_size, seq_length, d_model]
        return self.ln(output + residual)  # [batch_size, seq_length, d_model]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        ##
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len x batch_size x d_model]
        x = x + self.pe[:x.size(0), :]
        return x


class Encoder(nn.Module):
    def __init__(self):
        # The size is the lentgh of the input language
        self.src_emb = nn.Embedding(src_vocab_size, d_model).cuda()
        self.pos_emb = PositionalEncoding(d_model).cuda()
        self.layers = nn.ModuleList([EncoderLayer().cuda() for _ in range(n_layers)]).cuda()

    def forward(self, enc_input):
        #  enc_input: [batch_size x source_length]
        enc_outputs = self.src_emb(enc_input)  # [batch_size x source_length x d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0,
                                                                          1)  # [batch_size x source_length x d_model]
        enc_self_attention_mask = get_attn_pad_mask(enc_input,
                                                    enc_input).cuda()  # [batch_size x source_length x source_length]

        enc_self_attentions = []
        for layer in self.layers:
            enc_outputs, enc_self_attention = layer(enc_outputs, enc_self_attention_mask)
            enc_self_attentions.append(enc_self_attention)
        return enc_outputs, enc_self_attentions


class EncoderLayer(nn.Module):
    def __int__(self):
        super(EncoderLayer, self).__int__()
        self.enc_self_attention = MultiHeadAttention().cuda()
        self.pos_ffn = PoswiseFeedForwardNet().cuda()

    def forward(self, enc_inputs, enc_self_attention_mask):
        # enc_inputs: [batch_size x source_length x d_model]
        # enc_self_attention_mask: [batch_size x source_length x source_length]
        # enc_outputs: [batch_size x source_length x d_model]
        # enc_self_attention: [batch_size x source_length x source_length]
        # Q,K,V,MASKE
        enc_outputs, atten = self.enc_self_attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attention_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # [batch_size x source_length x d_model]
        return enc_outputs, atten


class DecoderLayer(nn.Module):
    def __int__(self):
        super(DecoderLayer, self).__int__()
        self.dec_self_attention = MultiHeadAttention().cuda()
        self.dec_enc_attention = MultiHeadAttention().cuda()
        self.pos_ffn = PoswiseFeedForwardNet().cuda()

    def forward(self, dec_inputs, enc_inputs, enc_outputs, dec_self_attention_mask, dec_enc_attention_mask):
        # dec_inputs: [batch_size x target_length x d_model]
        # enc_inputs: [batch_size x source_length x d_model]
        # enc_outputs: [batch_size x source_length x d_model]
        # dec_self_attention_mask: [batch_size x target_length x target_length]
        # dec_enc_attention_mask: [batch_size x target_length x source_length]
        # dec_outputs: [batch_size x target_length x d_model]
        # dec_self_attention: [batch_size x target_length x target_length]
        # dec_enc_attention: [batch_size x target_length x source_length]

        # Same source with Q,K,V
        dec_outputs, dec_self_attention = self.dec_self_attention(dec_inputs, dec_inputs, dec_inputs,
                                                                  dec_self_attention_mask)
        # Q: dec_outputs, K: enc_outputs, V: enc_outputs
        # The second attention layer
        dec_outputs, dec_enc_attention = self.dec_enc_attention(dec_outputs, enc_outputs, enc_outputs,
                                                                dec_enc_attention_mask)

        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attention, dec_enc_attention


class Decoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # The size is the lentgh of the output language
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model).cuda()
        self.pos_emb = PositionalEncoding(d_model).cuda()
        self.layers = nn.ModuleList([DecoderLayer().cuda() for _ in range(n_layers)]).cuda()

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        # dec_inputs: [batch_size x target_length]
        # enc_inputs: [batch_size x source_length]
        # enc_outputs: [batch_size x source_length x d_model]
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size x target_length x d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0,
                                                                          1)  # [batch_size x target_length x d_model]

        # Mask
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs,
                                                   dec_inputs).cuda()  # [batch_size x target_length x target_length]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(
            dec_inputs).cuda()  # [batch_size x target_length x target_length]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda()  # [batch_size x target_length x target_length]

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs,
                                              enc_inputs).cuda()  # [batch_size x target_length x source_length]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size x target_length x d_model]
            # dec_self_attn_mask: [batch_size x target_length x target_length]
            # dec_enc_attn_mask: [batch_size x target_length x source_length]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_input, dec_input):
        # enc_input: [batch_size x source_length]
        # dec_input: [batch_size x target_length]
        enc_output, enc_self_attention = self.encoder(enc_input)

        # dec_input: [batch_size x target_length x d_model]
        # dec_self_attention: [batch_size x target_length x target_length]
        # dec_enc_attention: [batch_size x target_length x source_length]
        dec_input, dec_self_attention, dec_enc_attention = self.decoder(dec_input, enc_input, enc_output)

        # dec_input: [batch_size x target_length x tgt_vocab_size]
        dec_logit = self.projection(dec_input)

        return dec_logit.view(-1, dec_logit.size(-1)), enc_self_attention, dec_self_attention, dec_enc_attention
