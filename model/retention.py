import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SeqSelfAttention(nn.Module):
    ATTENTION_TYPE_ADD = 'additive'
    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 **kwargs):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)
        self.Wt = torch.nn.Parameter(torch.Tensor(64, self.units))
        self.Wx = torch.nn.Parameter(torch.Tensor(64, self.units))
        self.Wa = torch.nn.Parameter(torch.Tensor(self.units, self.units))
        self.bh = torch.nn.Parameter(torch.Tensor(self.units)) if use_additive_bias else None
        self.ba = torch.nn.Parameter(torch.Tensor(1)) if use_attention_bias else None

    def _call_additive_emission(self, inputs):
        batch_size, input_len, _ = inputs.size()

        # h_{t, t'} = tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = torch.matmul(inputs, self.Wt).unsqueeze(2)
        k = torch.matmul(inputs, self.Wx).unsqueeze(1)

        if self.use_additive_bias:
            h = F.tanh(q + k + self.bh)
        else:
            h = F.tanh(q + k)

            # e_{t, t'} = W_a h_{t, t'} + b_a
        e = torch.matmul(h.view(batch_size * input_len * input_len, -1), self.Wa)
        if self.use_attention_bias:
            e += self.ba
        e = e.view(batch_size, input_len, input_len)
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}

    def forward(self, inputs, attention_width, mask=None, **kwargs):
        input_len = inputs.shape[1]
        e = self._call_additive_emission(inputs)
        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if attention_width is not None:
            lower = torch.arange(0, input_len) - attention_width // 2
            lower = lower.unsqueeze(-1)
            upper = lower + attention_width
            indices = torch.arange(0, input_len).unsqueeze(0)
            mask_lower = lower <= indices
            mask_upper = indices < upper
            mask = mask_lower & mask_upper
            mask_float = mask.float().to(e)
        max, _ = torch.max(e, dim=-1)
        e = torch.exp(e - max.unsqueeze(-1))

        # a_{t} = \text{softmax}(e_t)
        s = torch.sum(e, dim=-1).unsqueeze(-1)
        a = e / (s + 1e-8)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = torch.bmm(a, inputs)

        if self.return_attention:
            return [v, a]
        return v