
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

class GAT(torch.nn.Module):
    def __init__(self, type_in_channels, hidden_channels, out_channels, num_layers, dropout_p, num_head = 8, use_gpu=True, bias=True, use_multiple_gpu=False):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.use_gpu = use_gpu
        self.use_multiple_gpu = use_multiple_gpu

        self.transform_lins = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        if self.use_gpu is True:
            device = torch.device('cuda:0')
            for index in range(len(type_in_channels)):
                self.transform_lins.append(torch.nn.Linear(type_in_channels[index],hidden_channels,bias=False).to(device))
            for layer in range(num_layers):
                if use_multiple_gpu is True:
                    device = f"cuda:{layer % torch.cuda.device_count()}"
                else:
                    device = "cuda:0"
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_head, dropout=dropout_p, concat=False, bias=bias).to(device))
            layer = layer + 1
            if use_multiple_gpu is True:
                device = f"cuda:{layer % torch.cuda.device_count()}"
            else:
                device = "cuda:0"
            self.lin = torch.nn.Linear(hidden_channels*2,out_channels,bias=False).to(device)
        else:
            for index in range(len(type_in_channels)):
                self.transform_lins.append(torch.nn.Linear(type_in_channels[index],hidden_channels,bias=False))
            for _ in range(num_layers):
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_head, dropout=dropout_p, concat=False, bias=bias))
            self.lin = torch.nn.Linear(hidden_channels*2,out_channels,bias=False)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, all_init_mats, adjs, link, n_id, all_sorted_indexes, return_attention=False):

        if return_attention:
            self.attention_weights_list = []

        if self.use_gpu is True:

            x = torch.vstack([self.transform_lins[index](mat.to('cuda:0')) for index, mat in enumerate(all_init_mats)])[all_sorted_indexes][n_id]
            for i, (edge_index, size) in enumerate(adjs):
                if self.use_multiple_gpu is True:
                    device = f"cuda:{i % torch.cuda.device_count()}"
                else:
                    device = "cuda:0"
                x = x.to(device)
                x_target = x[:size[1]].to(device)
                edge_index = edge_index.to(device)
                if return_attention:
                    x, attention_weights = self.convs[i]((x, x_target), edge_index, return_attention_weights=True)
                    attention_weights = attention_weights.cpu().detach()
                    self.attention_weights_list += [(attention_weights[0],torch.mean(attention_weights[1],dim=1))]
                else:
                    x = self.convs[i]((x, x_target), edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
            i = i + 1
            if self.use_multiple_gpu is True:
                device = f"cuda:{i % torch.cuda.device_count()}"
            else:
                device = "cuda:0"
            if link.shape[0]==1:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]]].view(1,-1),x[[torch.where(n_id==i)[0][0] for i in link[:,1]]].view(1,-1)], dim=1)
            else:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]],:],x[[torch.where(n_id==i)[0][0] for i in link[:,1]],:]], dim=1)
            x = x.to(device)
#             x = torch.sigmoid(self.lin(x)).squeeze(1)
            x = self.lin(x).squeeze(1) 

        else:
            x = torch.vstack([self.transform_lins[index](mat) for index, mat in enumerate(all_init_mats)])[all_sorted_indexes][n_id]
            for i, (edge_index, size) in enumerate(adjs):
                x_target = x[:size[1]]
                if return_attention:
                    x, attention_weights = self.convs[i]((x, x_target), edge_index, return_attention_weights=True)
                    attention_weights = attention_weights.cpu().detach()
                    self.attention_weights_list += [(attention_weights[0],torch.mean(attention_weights[1],dim=1))]
                else:
                    x = self.convs[i]((x, x_target), edge_index)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            if link.shape[0]==1:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]]].view(1,-1),x[[torch.where(n_id==i)[0][0] for i in link[:,1]]].view(1,-1)], dim=1)
            else:
                x = torch.cat([x[[torch.where(n_id==i)[0][0] for i in link[:,0]],:],x[[torch.where(n_id==i)[0][0] for i in link[:,1]],:]], dim=1)
#             x = torch.sigmoid(self.lin(x)).squeeze(1)
            x = self.lin(x).squeeze(1)   

        return x
