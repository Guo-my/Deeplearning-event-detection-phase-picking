import torch
import torch.nn.functional as F


class AdditiveEmission(torch.nn.Module):
    def __init__(self, input_dim, use_additive_bias=True, use_attention_bias=True, attention_width=3):
        super(AdditiveEmission, self).__init__()
        self.input_dim = input_dim
        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.attention_width = attention_width

        # 初始化权重和偏置项
        self.Wt = torch.nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.Wx = torch.nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.Wa = torch.nn.Parameter(torch.Tensor(input_dim, 1))
        if use_additive_bias:
            self.bh = torch.nn.Parameter(torch.Tensor(input_dim))
        if use_attention_bias:
            self.ba = torch.nn.Parameter(torch.Tensor(1))

            # 重置参数
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.Wt)
        torch.nn.init.xavier_uniform_(self.Wx)
        torch.nn.init.xavier_uniform_(self.Wa)
        if self.use_additive_bias:
            torch.nn.init.zeros_(self.bh)
        if self.use_attention_bias:
            torch.nn.init.zeros_(self.ba)

    def forward(self, inputs):
        batch_size, input_len, _ = inputs.size()

        # h_{t, t'} = tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = torch.matmul(inputs, self.Wt).unsqueeze(2)  # (batch_size, input_len, 1, input_dim)
        k = torch.matmul(inputs, self.Wx).unsqueeze(1)  # (batch_size, 1, input_len, input_dim)
        h = q + k  # (batch_size, input_len, input_len, input_dim)
        if self.use_additive_bias:
            h += self.bh.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Broadcast bias to match dimensions of h
        h = torch.tanh(h)  # Apply tanh activation function

        # e_{t, t'} = W_a h_{t, t'} + b_a
        e = torch.matmul(h.view(batch_size * input_len * input_len, self.input_dim), self.Wa).view(batch_size,
                                                                                                   input_len,
                                                                                                   input_len)  # (batch_size, input_len, input_len)
        if self.use_attention_bias:
            e += self.ba.unsqueeze(-1).unsqueeze(-1)  # Broadcast bias to match dimensions of e
        e = torch.exp(e - torch.max(e, dim=-1, keepdim=True)[0])

        if self.attention_width is not None:
            lower = torch.arange(0, input_len) - self.attention_width // 2
            lower = lower.view(1, -1, 1).expand(batch_size, -1, input_len)
            upper = lower + self.attention_width
            indices = torch.arange(0, input_len).view(1, 1, -1).expand(batch_size, input_len, -1)
            mask_local = (lower <= indices) & (indices < upper)
            e = e * mask_local.float().to(e)
        s = torch.sum(e, dim=-1, keepdim=True)
        a = e / (s + 1e-8)

        v = torch.bmm(a, inputs)
        return v
