import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


class PositionalEncoding(nn.Module):
    def __init__(self, d_x, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_x)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_x, 2) * (-1 * math.log(10000) / d_x))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]].requires_grad_(False)

        return self.dropout(x)

    
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.s = math.sqrt(embedding_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len, dropout)
        
    def forward(self, x):
        x = self.word_embedding(x)
        x = self.pos_encoding(x * self.s)
        
        return x


class Attention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if exists(mask):
            max_neg_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(~mask, max_neg_value)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        
        return torch.matmul(p_attn, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_x, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_x, d_model)
        self.w_k = nn.Linear(d_x, d_model)
        self.w_v = nn.Linear(d_x, d_model)
        self.attention = Attention(dropout)
        self.w_o = nn.Linear(d_model, d_x)

    def forward(self, q, k, v, mask=None):
        if exists(mask):
            mask = mask.unsqueeze(1)     
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, self.d_k).transpose(1, 2) for x in (q, k, v)]
        z = self.attention(q, k, v, mask)
        z = z.transpose(1, 2).reshape(z.shape[0], -1, self.num_heads*self.d_k)
        z = self.w_o(z)

        return z


class AddNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):

        return self.norm(x + y)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_x, ffn_num_hiddens, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_x, ffn_num_hiddens)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(ffn_num_hiddens, d_x)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_x, d_model, ffn_num_hiddens, num_heads=8, dropout=0.1):
        super().__init__()
        self.atten = MultiHeadAttention(d_x, d_model, num_heads, dropout) 
        self.add_norm_1 = AddNorm(d_x)
        self.ffn = PositionwiseFeedForward(d_x, ffn_num_hiddens, dropout)
        self.add_norm_2 = AddNorm(d_x)

    def forward(self, x, mask=None):
        x = self.add_norm_1(x, self.atten(x, x, x, mask))

        return self.add_norm_2(x, self.ffn(x))


class TransformerEncoder(nn.Module):
    def __init__(self, d_x, d_model, ffn_num_hiddens, num_heads=8, num_blocks=6, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module("block" + str(i), EncoderBlock(d_x, d_model, ffn_num_hiddens, num_heads, dropout))

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_x, d_model, ffn_num_hiddens, num_heads=8, dropout=0.1):
        super().__init__()
        self.atten_1 = MultiHeadAttention(d_x, d_model, num_heads, dropout)
        self.add_norm_1 = AddNorm(d_x)
        self.atten_2 = MultiHeadAttention(d_x, d_model, num_heads, dropout)
        self.add_norm_2 = AddNorm(d_x)
        self.ffn = PositionwiseFeedForward(d_x, ffn_num_hiddens, dropout)
        self.add_norm_3 = AddNorm(d_x)

    def forward(self, x, memory, tgt_mask=None, src_mask=None): # memory = encoder output
        x = self.add_norm_1(x, self.atten_1(x, x, x, tgt_mask))
        x = self.add_norm_2(x, self.atten_2(x, memory, memory, src_mask))

        return self.add_norm_3(x, self.ffn(x))


class TransformerDecoder(nn.Module):
    def __init__(self, d_x, d_model, ffn_num_hiddens, num_heads=8, num_blocks=6, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module("block" + str(i), DecoderBlock(d_x, d_model, ffn_num_hiddens, num_heads, dropout))

    def forward(self, x, memory, tgt_mask=None, src_mask=None): # memory = encoder_output
        for block in self.blocks:
            x = block(x, memory, tgt_mask, src_mask)

        return x


def create_mask(src, tgt, pad=0): # src (b, l1) tgt (b, l2) 
    src_pad = src != pad
    src_pad_mask = src_pad.unsqueeze(1).expand(-1, src.shape[1], -1)
    tgt_ency_attn_mask = src_pad.unsqueeze(1).expand(-1, tgt.shape[1], -1)

    tgt_pad = tgt != pad
    tgt_pad_mask = tgt_pad.unsqueeze(1).expand(-1, tgt.shape[1], -1)

    tgt_seq_mask = torch.triu(torch.ones((tgt.shape[1], tgt.shape[1])), diagonal=1) == 0
    tgt_seq_mask = tgt_seq_mask.unsqueeze(0).expand(tgt.shape[0], -1, -1).type_as(tgt)

    tgt_self_attn_mask = tgt_seq_mask & tgt_pad_mask

    return src_pad_mask.bool(), tgt_self_attn_mask.bool(), tgt_ency_attn_mask.bool()


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embedding_dim, d_model, ffn_num_hiddens, max_len=5000, num_heads=8, num_blocks=6, dropout=0.1):
        super().__init__()
        self.src_emb = Embedding(src_vocab_size, embedding_dim, max_len, dropout)
        self.tgt_emb = Embedding(tgt_vocab_size, embedding_dim, max_len, dropout)
        self.encoder = TransformerEncoder(embedding_dim, d_model, ffn_num_hiddens, num_heads, num_blocks, dropout)
        self.decoder = TransformerDecoder(embedding_dim, d_model, ffn_num_hiddens, num_heads, num_blocks, dropout)
        self.proj = nn.Linear(embedding_dim, tgt_vocab_size)

    def forward(self, src, tgt, src_pad_mask=None, tgt_self_attn_mask=None, tgt_ency_attn_mask=None):
        src = self.src_emb(src)
        tgt = self.tgt_emb(tgt)
        memory = self.encoder(src, src_pad_mask)
        output = self.decoder(tgt, memory, tgt_self_attn_mask, tgt_ency_attn_mask)

        return self.proj(output)
 
