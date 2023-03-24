import torch
import math

def special_quantize_grad_input(grad_output, q_weight, scale_weight, num_bits, q_input):
    assert grad_output.dtype == torch.float16, "grad_output must be half float datatype!"
    # assert dequantize_weight.dtype == torch.float16, "dequantize_weight must be half float datatype!"
    # assert h_input.dtype == torch.float16, "h_input must be half float datatype!"
    mn = min(grad_output.min() - 1e-8, 0).float()
    mx = max(grad_output.max() + 1e-8, 0).float()
    
    zero_point1 = mn
    scale1 = num_bins / (mx - mn)
    
    qzero = -zero_point1 * scale1
    iqzero = torch.floor(qzero)
    
    if iqzero > 0:
        mx = (iqzero - num_bins) * mn / iqzero
    elif iqzero == 0:
        zero_point1, mn = 0, 0

    scale1 = num_bins / (mx - mn)
    
    first_transform = (grad_output.float() - zero_point1) * scale1 - 8
    first_transform.clamp_(-8.0, num_bins-8).round_()
    first_quantize = ((first_transform+8) / scale1 + zero_point1).half()
    
    residual = grad_output - first_quantize
    
    mn = min(residual.min() - 1e-8, 0).float()
    mx = max(residual.max() + 1e-8, 0).float()
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output.float() * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    first_quantize = (first_transform * num_bins_half / scale1).half()
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    second_quantize = (second_transform / scale1).half()
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    
    # leverage score
    x_len = torch.linalg.norm(output_dequantize, dim=1)
    vec_norm = x_len.float()
    len_norm = len(vec_norm)
        
    cnt = 0
    norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum())

    while norm_activation_loop.max() > 1 and cnt < len_norm / 2:
        small_index = (norm_activation_loop < 1)
        small_value = norm_activation_loop[small_index]
        cnt = len_norm - len(small_value)
        norm_activation_loop = torch.clamp(norm_activation_loop, 0, 1)
        if small_value.max() == 0:
            break
        small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
        norm_activation_loop[small_index] = small_value
    
    sample_index = torch.bernoulli(norm_activation_loop)
    left_indices = (sample_index != 1)
    norm_activation_loop[norm_activation_loop == 0] = 1e-10
    output_dequantize =  output_dequantize / norm_activation_loop.unsqueeze(1)
    output_dequantize[left_indices] = 0
    
    # dequantize inputx
    dequantize_sample_x = (output_dequantize[0:output_dequantize.shape[0] // 2] + output_dequantize[output_dequantize.shape[0] // 2:]).half()
    # dequantize_sample_x = first + second
    
    # dequantize inputy
    dequantize_sample_y = q_weight * (scale_weight.half())
    grad_out = dequantize_sample_x.matmul(dequantize_sample_y)
    
    # calculate grad_activation and grad_scale_activation through LSQ
    # q_w = h_input / scale_input
    indicate_small = (q_input < -8).half()
    indicate_big = (q_input > 7).half()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(q_input.numel() * 7)
    grad_scale_input = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -q_input + q_input.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_input = indicate_middle * grad_out
    
    return grad_input, grad_scale_input

def special_quantize_grad_weight(grad_output, q_input, scale_input, num_bits, q_w):
    assert grad_output.dtype == torch.float16, "grad_output must be half float datatype!"
    # assert dequantize_input.dtype == torch.float16, "dequantize_input must be half float datatype!"
    mn = min(grad_output.min() - 1e-8, 0).float()
    mx = max(grad_output.max() + 1e-8, 0).float()
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output.float() * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    first_quantize = (first_transform * num_bins_half / scale1).half()
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    second_quantize = (second_transform / scale1).half()
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    
    dequantize_input = q_input * (scale_input.half())
    y2 = torch.cat([dequantize_input, dequantize_input], 0)
    x_len = torch.linalg.norm(output_dequantize, dim=1)
    y_len = torch.linalg.norm(y2, dim=1)
    
    vec_norm = x_len.mul(y_len).float()
    len_norm = len(vec_norm)
    
    cnt = 0
    norm_weight_loop = vec_norm * len_norm / (2 * vec_norm.sum())

    while norm_weight_loop.max() > 1 and cnt < len_norm / 2:
        small_index = (norm_weight_loop < 1)
        small_value = norm_weight_loop[small_index]
        cnt = len_norm - len(small_value)
        norm_weight_loop = torch.clamp(norm_weight_loop, 0, 1)
        if small_value.max() == 0 :
            break
        small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
        norm_weight_loop[small_index] = small_value

    sample_index = torch.bernoulli(norm_weight_loop)
    index = torch.nonzero((sample_index == 1)).squeeze(1)
    norm_weight_loop[norm_weight_loop == 0] = 1e-10
    output_dequantize = output_dequantize / norm_weight_loop.unsqueeze(1)
    
    sample_x = output_dequantize[index, :]
    sample_y = y2[index, :]
    
    # dequantize inputx    
    dequantize_sample_x = sample_x.half()
    
    # dequantize inputy
    dequantize_sample_y = sample_y 
    grad_out = dequantize_sample_x.t().matmul(dequantize_sample_y)
    
    indicate_small = (q_w < -8).half()
    indicate_big = (q_w > 7).half()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(q_w.numel() * 7)
    grad_scale_weight = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -q_w + q_w.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_weight = indicate_middle * grad_out
    
    return grad_weight, grad_scale_weight