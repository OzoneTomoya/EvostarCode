import torch

def calc_mean_bit(layer_bits):
    mean = sum(layer_bits.values()) / len(layer_bits)
    return mean
    
def Initialization(bit):
    layer_bits = {i: bit for i in range(32)}
    
    return layer_bits

def get_minq_maxq(bits):
    maxq = 2**bits - 1
    minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, minq, maxq):
    x_fp32 = x.float()
    q = torch.clamp(torch.round(x_fp32 / scale) + zero, minq, maxq)
    return q

def asym_dequant(q, scale, zero):
    return (q - zero).float() * scale

def find_param(tensor, minq, maxq):
    tensor_fp32 = tensor.float()
    x_min=tensor_fp32.min()
    x_max=tensor_fp32.max()

    scale = (x_max - x_min) / (maxq - minq)
    scale = max(scale, 1e-5)

    zero = torch.round(minq - x_min / scale)

    return scale, zero

def quant_dequant(tensor, bits):
    minq, maxq = get_minq_maxq(bits)
    scale, zero = find_param(tensor, minq, maxq)

    q_tensor = asym_quant(tensor, scale, zero, minq, maxq)
    deq_tensor = asym_dequant(q_tensor, scale, zero)

    return deq_tensor.to(dtype=tensor.dtype)

def Quantization(model, layer_bits):
    
    for i, layer in enumerate(model.model.layers): 
        bits = layer_bits.get(i, 8)
        minq, maxq = get_minq_maxq(bits)

        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            linear = getattr(layer.self_attn, proj)
            with torch.no_grad():
                linear.weight.data = quant_dequant(linear.weight.data, bits)

        for proj in ['up_proj', 'gate_proj', 'down_proj']:
            linear = getattr(layer.mlp, proj)
            with torch.no_grad():
                linear.weight.data = quant_dequant(linear.weight.data, bits)