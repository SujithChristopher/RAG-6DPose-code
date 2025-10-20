import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, queries, keys, values, mask = None):
        b, n, _, h = *queries.shape, self.heads
        
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        
        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
            
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        
        out = out.transpose(1, 2).contiguous().view(b, n, -1).contiguous()
        return out


class SurfAttention(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
    def forward(self, queries, keys, values, mask = None):
        b, n, _, h = *queries.shape, self.heads
        
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        
        queries = queries.view(b, n, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale
        
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, n, -1)
            dots.masked_fill_(~mask, float('-inf'))
            
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        
        out = out.transpose(1, 2).contiguous().view(b, n, -1).contiguous()
        return out

class SelfAttention(nn.Module):
    def __init__(self, dim, fusion_dim , heads = 8):
        super().__init__()
        self.dim = dim
        self.fusion_dim = fusion_dim
        self.heads = heads
        self.scale = ((dim*fusion_dim) ** -0.5) ** -0.5
        
        self.query = nn.Linear(dim, fusion_dim)
        self.key = nn.Linear(dim, fusion_dim)
        self.value = nn.Linear(dim, fusion_dim)
        
    def forward(self, queries, keys, values, mask = None):
        n, _, h = *queries.shape, self.heads
        
        queries = self.query(queries)
        keys = self.key(keys)
        values = self.value(values)
        
        queries = queries.view(n, h, -1).transpose(0, 1).contiguous()
        keys = keys.view(n, h, -1).transpose(0, 1).contiguous()
        values = values.view(n, h, -1).transpose(0, 1).contiguous()
        
        dots = torch.einsum('hid,hjd->hij', queries, keys) * self.scale
        

            
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('hij,hjd->hid', attn, values)
        
        out = out.transpose(0, 1).contiguous().view(n, -1).contiguous()
        return out
    



class SelfAttention_Feature_level(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()

        self.query_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_matrix = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(0.1)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        """
        """
        batch_size, seq_len, hidden_dim = x.size()

        Q = self.query_matrix(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key_matrix(x)  # (batch_size, seq_len, hidden_dim)
        V = self.value_matrix(x)  # (batch_size, seq_len, hidden_dim)

        scores = torch.matmul(Q, K.transpose(1, 2))  # (batch_size, seq_len, seq_len)

        scaled_scores = scores / self.scale  # (batch_size, seq_len, seq_len)

        attn_weights = torch.softmax(scaled_scores, dim=-1)  # (batch_size, seq_len, seq_len)

        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, hidden_dim)

        return attn_output, attn_weights

