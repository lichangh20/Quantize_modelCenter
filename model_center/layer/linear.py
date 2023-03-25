# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import bmtrain as bmt
import math
import torch.nn.functional as F
from torch import nn
import time
import quantize_forward_easy as qfe
import quantize_grad_weight_speed as qgw
import quantize_grad_input_speed as qgi
from torch.autograd.function import Function
from .special_quantize import special_quantize_grad_input, special_quantize_grad_weight

class Linear(bmt.DistributedModule):
    r"""A fully connected layer, which performs :math:`\pmb{y} = \mathbf{W} \pmb{x} + \pmb{b}`

    Args:
        dim_in (int): input dimension of :math:`\pmb{x}`
        dim_out (int): output dimension of :math:`\pmb{y}`
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
        bias (bool, optional): whether to add bias term :math:`\pmb{b}`. Defaults to False.
    """
    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 length_scale : bool = False,
                 length_scale_before : bool = False,
                 dtype = torch.half,
                 int8 : bool = False,
                 init_mean : float = 0.0,
                 init_std : float = 1,
                 bias : bool = False,
                ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((dim_out,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer

        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.

        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        x = F.linear(x, self.weight)
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        if self.bias is not None:
            x = x + self.bias
        return x
    
class QuantizationConfig:
    def __init__(self):
        self.hadamard_group = 32
        self.forward = {}
        self.grad_weight = {}
        self.grad_input = {}
        self.hadamard = 0
        self.scale = 0
        self.special_layer = 0
        self.linear_forward = 0
        self.linear_backward = 0
        self.backward_step = {}
        self.initial = 0
        
qconfig = QuantizationConfig()

def time_forward(time_vector):
    key_list = ["quantize", "pack", "gemm", "dequantize"]
    for keyindex in range(len(key_list)):
        keyname = key_list[keyindex]
        if keyname not in qconfig.forward.keys():
            qconfig.forward[keyname] = time_vector[keyindex]
        else:
            qconfig.forward[keyname] += time_vector[keyindex]
            
def time_grad_weight(time_vector):
    key_list = ["quantize", "leverage", "sample", "pack", "gemm", "dequantize", "LSQ", "method1", "method2", "method3"]
    for keyindex in range(len(key_list)):
        keyname = key_list[keyindex]
        if keyname not in qconfig.grad_weight.keys():
            qconfig.grad_weight[keyname] = time_vector[keyindex]
        else:
            qconfig.grad_weight[keyname] += time_vector[keyindex]
            
def time_grad_input(time_vector):
    key_list = ["quantize", "leverage", "sample", "pack", "gemm", "dequantize" ,"LSQ"]
   
    for keyindex in range(len(key_list)):
        keyname = key_list[keyindex]
        if keyname not in qconfig.grad_input.keys():
            qconfig.grad_input[keyname] = time_vector[keyindex]
        else:
            qconfig.grad_input[keyname] += time_vector[keyindex]

class linear_act_cuda(Function):
    @staticmethod
    def forward(ctx, h_input, h_weight, scale_input, scale_weight, bias):
        # print("forward")
        input_shape = h_input.shape
        C_in = h_input.shape[-1]
        
        h_input_flatten = h_input.reshape(-1,C_in)
        # h_input_flatten = h_input.reshape(-1,C_in).half()
        # h_weight = h_weight.half()
        
        out_shape = list(h_input.shape)
        out_shape[-1] = h_weight.shape[0]
        out = qfe.quantize(h_input_flatten, h_weight, scale_input, scale_weight)
        # forward_time_vector = out[3]
        # time_forward(forward_time_vector)
        output = out[0].reshape(out_shape)
        
        # out[2] is q_input_flatten, out[3] is q_weight
        ctx.saved = out[1], out[2], scale_input, scale_weight, bias, input_shape
        
        if bias is not None:
            return output + bias
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # assert torch.isnan(grad_output).any() == False
        # print("backward")
        
        # torch.cuda.synchronize()
        # time_linear_start = time.time()
        
        C_out = grad_output.shape[-1]
        # assert grad_output.dtype == torch.float16
        grad_output_flatten = grad_output.reshape(-1, C_out)
        # grad_output_flatten = grad_output.reshape(-1, C_out).half()
        q_input_flatten, q_weight, scale_input, scale_weight, bias, input_shape = ctx.saved
        # dequantize_input_flatten = (q_input_flatten * scale_input.half())
        # dequantize_weight = (q_weight * scale_weight.half())
        
        flag_weight = (q_input_flatten.shape[1] % 4 != 0)
        if flag_weight:
            start = time.time()
            grad_weight, grad_scale_weight = special_quantize_grad_weight(grad_output_flatten, q_input_flatten, scale_input, 4, q_weight)
            torch.cuda.synchronize()
            qconfig.special_layer += time.time() - start
        else:
            # h_weight = (q_weight * scale_weight.half())
            # dequantize_input_flatten = (q_input_flatten * scale_input.half())
            # weight_out = quantize_grad_weight_speed.quantize(grad_output_flatten, 4, dequantize_input_flatten, q_input_flatten, scale_input, h_weight, scale_weight)
            
            weight_out = qgw.quantize(grad_output_flatten, 4, q_input_flatten, scale_input, q_weight)
            grad_weight, grad_scale_weight = weight_out[0], weight_out[1]
            # grad_weight_time_vector = weight_out[3]
            # time_grad_weight(grad_weight_time_vector)
            

        # then calculate grad_input_flatten and grad_scale_input
        flag_input = (grad_output_flatten.shape[1] % 32 != 0) or (q_weight.shape[1] % 4 != 0)
        if flag_input or flag_weight:
            start = time.time()
            grad_input_flatten, grad_scale_input = special_quantize_grad_input(grad_output_flatten, q_weight, scale_weight, 4, q_input_flatten)
            torch.cuda.synchronize()
            qconfig.special_layer += time.time() - start
        else:
            activation_out = qgi.quantize(grad_output_flatten, 4, q_weight, scale_weight, q_input_flatten, weight_out[-5], weight_out[-4], weight_out[-3], weight_out[-2], weight_out[-1])
            # suppose out1 is grad_input_flatten, out2 is grad_scale_input
            grad_input_flatten, grad_scale_input = activation_out[0], activation_out[1]
            
            # grad_input_time_vector = activation_out[4]
            # time_grad_input(grad_input_time_vector)
            
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
            
        grad_input = grad_input_flatten.reshape(input_shape)
        # torch.cuda.synchronize()
        # time_linear_end = time.time()
        # qconfig.linear_backward += time_linear_end - time_linear_start
        return grad_input, grad_weight, grad_scale_input, grad_scale_weight, grad_bias

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        # print(qconfig.hadamard_group)
        super(QLinear, self).__init__(in_features, out_features, bias)

        T = {}

        size = 1
        H = torch.ones(1, 1).cuda()
        T[1] = H

        for i in range(7):
            H = torch.cat((torch.cat([H, H], 1),
                           torch.cat([H, -H], 1)), 0) / math.sqrt(2)
            size *= 2
            T[size] = H

        self.initialize_weight = False
        self.initialize_input = False
        self.first_pass = False
        self.dim = self.weight.shape[-1]
        self.hadamard = T[qconfig.hadamard_group].half()
        self.hadamard_float = T[qconfig.hadamard_group] 
        self.weight.data = self.weight.data.half()
        # self.weight = bmt.DistributedParameter(
        #     torch.empty((out_features, in_features), dtype=torch.float16),
        #     init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        # )
        # self.bias = bmt.DistributedParameter(
        #     torch.empty((out_features,), dtype=torch.float16),
        #     init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        # ) if bias else None
        
    def forward(self, input : torch.Tensor):
        # torch.cuda.synchronize()
        # time_linear_start = time.time()
        
        if self.first_pass:
            self.scale_input.data = self.scale_input.data.abs()
            self.scale_weight.data = self.scale_weight.data.abs()
            
        if not self.first_pass:
            print("Actually Using QLinear!")
            self.first_pass = True
            
        # torch.cuda.synchronize()
        # time_hadamard_start = time.time()
        # qconfig.scale += time_hadamard_start - time_linear_start
        
        input_flatten = input.view(-1, input.shape[-1])
        input_shape = input.shape
        
        # print("input", input_flatten.abs().max())
        
        h_input = input_flatten.reshape(-1, qconfig.hadamard_group).matmul(self.hadamard).reshape(input_shape)
        # print("h_input", h_input.float().abs().max())
        # print("h_input2", h_input.float().abs().min())
        # print("h_input3", h_input.float().abs().mean())
        
        weight_shape = self.weight.shape
        h_weight = self.weight.reshape(-1, qconfig.hadamard_group).matmul(self.hadamard).reshape(weight_shape)
        
        # torch.cuda.synchronize()
        # time_hadamard_end = time.time()
        # qconfig.hadamard += time_hadamard_end - time_hadamard_start
        
        if not self.initialize_input:
            # print(h_input)
            self.scale_input = nn.Parameter(2 * h_input.float().abs().mean() / math.sqrt(7) + 1e-10, requires_grad=True)
            print("init", self.scale_input.min())
            self.initialize_input = True
            
        if not self.initialize_weight:
            # print(h_weight)
            self.scale_weight = nn.Parameter(2 * h_weight.float().abs().mean() / math.sqrt(7) + 1e-10, requires_grad=True)
            print("init", self.scale_weight.min())
            self.initialize_weight = True
            
        qbias = self.bias.half() if self.bias is not None else None
        
        output = linear_act_cuda.apply(h_input, h_weight, self.scale_input, self.scale_weight, qbias)
        
        # torch.cuda.synchronize()
        # time_linear_end = time.time()
        # qconfig.linear_forward += time_linear_end - time_linear_start
        return output
