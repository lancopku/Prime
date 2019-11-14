import torch.nn as nn
import math


def Linear(in_features, out_features, layer_id=0, args=None, cur_linear=None, bias=True, ):
    m = nn.Linear(in_features, out_features, bias)
    init_method = args.init_method if 'init_method' in args else 'xavier'
    if args is None:
        nn.init.xavier_uniform_(m.weight)
    else:
        if init_method == 'xavier':
            nn.init.xavier_uniform_(m.weight)
        elif init_method == 'fixup':
            nn.init.xavier_uniform_(m.weight,  gain=1/math.sqrt(6))
        elif init_method == 'xi':
            gain = (layer_id+1)**(-0.5)
            nn.init.xavier_uniform_(m.weight, gain=gain)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m