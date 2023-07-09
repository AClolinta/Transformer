
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


    class Encoder(nn.Module):
        def __init__(self) :
            # The size is the lentgh of the input language
            self.src_emb = nn.Embedding(src_vocab_size, d_model).cuda()
            self.pos_emb = PositionalEncoding(d_model).cuda()
            self.layers = nn.ModuleList([EncoderLayer().cuda() for _ in range(n_layers)]).cuda()

        def forward(self, enc_input):
            #  enc_input: [batch_size x source_length]
            enc_outputs = self.src_emb(enc_input) # [batch_size x source_length x d_model]
            enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1) # [batch_size x source_length x d_model]
            enc_self_attention_mask = get_attn_pad_mask(enc_input, enc_input).cuda() # [batch_size x source_length x source_length]

            enc_self_attentions = []
            for layer in self.layers :
                enc_outputs, enc_self_attention = layer(enc_outputs,enc_self_attention_mask)
                enc_self_attentions.append(enc_self_attention)
            return enc_outputs,enc_self_attention
        
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
            enc_outputs = self.pos_ffn(enc_outputs) # [batch_size x source_length x d_model]
            return enc_outputs, atten


    class Decoder(nn.Module):
        def __init__(self,d_model,dropout=0.1,max_len=5000) :
            # The size is the lentgh of the output language
            self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model).cuda()
            self.pos_emb = PositionalEncoding(d_model).cuda()
            self.layers = nn.ModuleList([DecoderLayer().cuda() for _ in range(n_layers)]).cuda()


