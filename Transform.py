
import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_input, dec_input):

        # enc_input: [batch_size x source_length]
        # dec_input: [batch_size x target_length]
        enc_output,enc_self_attention = self.encoder(enc_input)

        # dec_input: [batch_size x target_length x d_model]
        # dec_self_attention: [batch_size x target_length x target_length]
        # dec_enc_attention: [batch_size x target_length x source_length]
        dec_input,dec_self_attention,dec_enc_attention = self.decoder(dec_input, enc_input,enc_output)

        # dec_input: [batch_size x target_length x tgt_vocab_size]
        dec_logit = self.projection(dec_input)

        return dec_logit.view(-1, dec_logit.size(-1)), enc_self_attention, dec_self_attention, dec_enc_attention

    class Encoder(nn.Module):
        def __init__(self) :
            # The size is the lentgh of the input language
            self.src_emb = nn.Embedding(src_vocab_size, d_model).cuda()
            self.pos_emb = PositionalEncoding(d_model).cuda()
            self.layers = nn.ModuleList([EncoderLayer().cuda() for _ in range(n_layers)]).cuda()

    class Decoder(nn.Module):
        def __init__(self,d_model,dropout=0.1,max_len=5000) :
            # The size is the lentgh of the output language
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model).cuda()
            self.pos_emb = PositionalEncoding(d_model).cuda()
            self.layers = nn.ModuleList([DecoderLayer().cuda() for _ in range(n_layers)]).cuda()

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_seq_len = 200):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            ##
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:,0::2] = torch.sin(position * div_term)
            pe[:,1::2] = torch.cos(position * div_term)

            #
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            # x: [seq_len x batch_size x d_model]
            x = x + self.pe[:x.size(0), :]
            return x