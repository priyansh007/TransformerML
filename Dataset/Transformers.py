import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
    def forward(self):
        even_i = torch.arange(0,self.d_model,2,dtype=float)
        pos = torch.arange(0, self.seq_len, 1, dtype=float).reshape(self.seq_len, 1)
        denominator = torch.pow(10000, even_i/self.d_model)
        odd_PE = torch.sin(pos/denominator)
        even_PE = torch.cos(pos/denominator)
        final = torch.stack([even_PE,odd_PE],dim=2)
        final = torch.flatten(final, start_dim=1, end_dim=2)
        return final
    
def scaled_dot_products(q,k,v,mask=None):
    #q,k,v each 30 x 8 x 200 x 64
    d_k = q.size()[-1] #64
    scaled = torch.matmul(q,k.transpose(-1,-2)) / math.sqrt(d_k) # 30 x 8 x 200 x 200
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1) # Matrix of no. seq X no. seq which have self attention and cross attention 30 x 8 x 200 x 200 - Mask seq x seq
    values = torch.matmul(attention, v) #30 x 8 x 200 x 64
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model #512
        self.num_heads = num_heads #8
        self.head_dim = d_model // num_heads #64
        self.qkvlayer = nn.Linear(d_model, 3*d_model) #512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model) #512 x 512
    
    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.shape # 30 x 200 x 512
        qkv = self.qkvlayer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim) #30 x 200 x 8 x 192
        qkv = qkv.permute(0,2,1,3) #30 x 8 x 200 x 192
        q, k ,v = qkv.chunk(3, dim=-1) # each 30 x 8 x 200 x 64
        values, attention = scaled_dot_products(q,k,v,mask) #30 x 8 x 200 x 64
        values = values.reshape(batch_size, seq_length, self.num_heads*self.head_dim) #30 x 200 x 512
        out = self.linear_layer(values) #30 x 200 x 512
        return values

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameter_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
    
    def forward(self, inputs):
        dims = [-(i+1) for i in range(len(self.parameter_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs-mean) ** 2).mean(dim=dims, keepdim=True) #30x200x1
        std = (var+self.eps).sqrt()
        y = (inputs-mean)/std
        return self.gamma * y + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x 

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.dropout = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
    def forward(self, x):
        resid_x = x
        x = self.attention(x)
        x = self.dropout(x)
        x = self.norm(resid_x+x)
        resid_x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm(x+resid_x)
        return x   

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers_encoder):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers_encoder)])
        
    def forward(self, x):
        x = self.layers(x)
        return x     

class MultiHeadAttentionSecond(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model #512
        self.num_heads = num_heads #8
        self.head_dim = d_model // num_heads #64
        self.q_layer = nn.Linear(d_model, d_model)
        self.kvlayer = nn.Linear(d_model, 2*d_model) #512 x 1024
        self.linear_layer = nn.Linear(d_model, d_model) #512 x 512
    
    def forward(self, x, encoder_output):
        batch_size, seq_length, d_model = x.shape # 30 x 200 x 512
        kv = self.kvlayer(encoder_output) # 30 x 200 x 1536
        q = self.q_layer(x)
        kv = kv.reshape(batch_size, seq_length, self.num_heads, 2*self.head_dim) #30 x 200 x 8 x 128
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        kv = kv.permute(0,2,1,3) #30 x 8 x 200 x 128
        q = q.permute(0,2,1,3)
        k ,v = kv.chunk(2, dim=-1) # each 30 x 8 x 200 x 64
        values, attention = scaled_dot_products(q,k,v,None) #30 x 8 x 200 x 64
        values = values.reshape(batch_size, seq_length, self.num_heads*self.head_dim) #30 x 200 x 512
        out = self.linear_layer(values) #30 x 200 x 512
        return values

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttentionSecond(d_model, num_heads)
        self.norm = LayerNormalization(parameters_shape=[d_model])
        self.dropout = nn.Dropout(p=drop_prob)
        self.ffn = FeedForward(d_model, ffn_hidden, drop_prob)
    def forward(self, x, encoder_output, decoder_mask):
        resid_x = x
        x = self.attention(x, decoder_mask)
        x = self.dropout(x)
        x = self.norm(resid_x+x)
        resid_x = x
        x = self.attention2(x, encoder_output)
        x = self.dropout(x)
        x = self.norm(resid_x+x)
        resid_x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm(x+resid_x)
        return x 

class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x,encoder_output,mask = inputs
        for module in self._modules.values():
            x = module(x,encoder_output,mask)
        return x
class Decoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers_decoder):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                     for _ in range(num_layers_decoder)])
        
    def forward(self, x, encoder_output, mask):
        x = self.layers(x, encoder_output, mask)
        return x  
    
#Example
d_model = 512
num_heads = 8
max_seq_len = 200
batch_size = 30

num_layers_encoder=5
num_layers_decoder=6

ffn_hidden_layers = 2048
dropout = 0.1

tensor_English = torch.randn((batch_size, max_seq_len, d_model)).float()
tensor_Gujarati = torch.randn((batch_size, max_seq_len, d_model)).float()
mask = torch.full([max_seq_len, max_seq_len] , float('-inf'))
mask = torch.triu(mask, diagonal=1)

encoder = Encoder(d_model, ffn_hidden_layers, num_heads, dropout, num_layers_encoder)
decoder = Decoder(d_model, ffn_hidden_layers, num_heads, dropout, num_layers_decoder)
single_layer = nn.Linear(d_model,d_model)
positional_encoding = PositionalEncoding(max_seq_len,d_model)

encoder_positional = positional_encoding.forward()
input_Encoder = single_layer(tensor_English) 
output_encoder = encoder.forward(input_Encoder+encoder_positional.float())

input_Decoder = single_layer(tensor_Gujarati)
output_Decoder = decoder.forward((input_Decoder+encoder_positional.float()), output_encoder, mask)