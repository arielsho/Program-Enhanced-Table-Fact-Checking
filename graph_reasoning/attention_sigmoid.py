import torch
import torch.nn as nn

class AttentionLayer_sigmoid(nn.Module):
    def __init__(self, args):
        super(AttentionLayer_sigmoid, self).__init__()
        self.args = args
        self.use_gpu = True
        self.dim1 = self.args.in_dim
        self.dim2 = self.args.mem_dim
        self.hidden_dim = self.args.mem_dim
        self.linear1 = nn.Linear(self.args.in_dim, self.hidden_dim, bias=False)
        self.linear2 = nn.Linear(self.args.in_dim, self.hidden_dim, bias=True)
        self.v = nn.Linear(self.hidden_dim, 1, bias=False)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        if self.use_gpu:
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()
            self.v = self.v.cuda()
            self.sm = self.sm.cuda()
            self.tanh = self.tanh.cuda()

    def attention_weight(self, h_t, h_s):
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        wq = self.linear1(h_t.contiguous().view(-1, self.dim1))
        wq = wq.view(tgt_batch, tgt_len, self.hidden_dim)
        uh = self.linear2(h_s)
        uh = uh.view(tgt_batch, 1, self.hidden_dim).expand(tgt_batch, tgt_len, self.hidden_dim)
        wquh = self.tanh(wq + uh)
        res = torch.sigmoid(self.v(wquh.contiguous().view(-1, self.hidden_dim)).view(tgt_batch, tgt_len))

        return res

    def forward(self, input, memory_bank, memory_mask=None):
        batch_, tgtL, tgt_dim = memory_bank.size()
        align = self.attention_weight(memory_bank, input).unsqueeze(dim=2)
        if memory_mask is not None:
            align = align * memory_mask.unsqueeze(dim=2).float()

        return torch.sum(align.expand(batch_, tgtL, tgt_dim) * memory_bank, dim=1)

