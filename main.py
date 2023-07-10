import Transform
import torch
import torch.nn as nn

###########Data###########
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]
enc_inputs, dec_inputs, dec_outputs = Transform.make_data(sentences)

# Padding Should be Zero
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, '.': 6, 'S': 7, 'E': 8}
tgt_vocab_size = len(tgt_vocab)

src_len = 5  # enc_input max sequence length
tgt_len = 6  # dec_input(=dec_output) max sequence length

loader = Transform.loader
################### Hyper Parameters ###################

d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

###########Model###########

model = Transform.Transformer().cuda()
crit = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(50):
    # enc_inputs: [batch_size, src_len]
    # dec_inputs: [batch_size, tgt_len]
    # dec_outputs: [batch_size, tgt_len]
    for enc_inputs,dec_inputs,dec_outputs in loader:
        enc_inputs = enc_inputs.cuda()
        dec_inputs = dec_inputs.cuda()
        dec_outputs = dec_outputs.cuda()

        # enc_outputs: [batch_size, src_len, d_model]
        # enc_self_attns: [batch_size, n_heads, src_len, src_len]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = crit(outputs, dec_outputs.view(-1))

        # log
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        # model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test
enc_inputs,_,_ = next(iter(loader))
enc_inputs = enc_inputs.cuda()

for i in range(len(enc_inputs)):
    greedy_dec_input = Transform.greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [Transform.idx2word[n.item()] for n in predict.squeeze()])


