import torch
import numpy as np
import torch.nn as nn
seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)



class MyLayer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1, tile = 1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError('Only groupsize -1 and 128 are supported.')
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError('`infeatures` must be divisible by 128 and `outfeatures` by 256.')
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError('`infeatures` must be divisible by `groupsize`.')
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer('B', torch.empty((self.n * tile // 8 , self.k // tile), dtype=torch.int))
        self.register_buffer('s', torch.empty((self.k // groupsize, self.n), dtype=torch.half))

    def pack(self, linear, scales, tile):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """ 
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        maxq = 2 ** 4 - 1
        s = scales.t()
        w = linear.weight.data

        w = torch.round(w / s.T).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        
        k = self.k
        
        interleave = []
        for i in range(k//8):
            out = [0, 2, 4, 6, 1, 3, 5, 7]
            for j in range(8):
                out[j] = out[j] + 8 * i
            interleave += out
        interleave = np.array(interleave)
        # print(interleave)
        # exit()
        # s = s.reshape((-1, len(interleave)))[:, interleave]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.n // tile, tile, self.k // tile, tile))
        
        # w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.n * tile, self.k // tile))
        # print(w.shape)
        # print(w)
        
        
        res = w[:,interleave]
      
        # res = res.reshape((-1, _perm.numel()))[:, interleave].reshape(res.shape)
        # print("perm is ")
        # print(_scale_perm_single)
        # print(_perm)
        # exit()
      
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
   
        res = res.cpu().numpy().astype(np.uint32)
        
        #print( np.sum(res == 8) / (res.shape[0] * res.shape[1]) )
        # exit()
        # print(res.shape)

        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        # print(q.shape)
        # exit()
        # print("target shape")
        # print(self.B.shape)
        # exit()
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)


maxq = 2 ** 4 - 1


def gen_quant4_my(n, k, w, groupsize=-1,  tile = 1):
   
    DEV = w.device
    #print(w)
    s = torch.max(torch.abs(w), 1, keepdim=True)[0].to(w.device)
    s *= 2 / maxq
 
  
    #print(torch.max(torch.abs(ref), 0, keepdim=True)[0])
    #exit()

    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n).to(torch.float16).cuda()
    linear.weight.data.copy_(w)

    
    # Workaround to test some special cases that are forbidden by the API
    layer = MyLayer(256, 256, groupsize=groupsize, tile = tile)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((n // tile , k  * tile // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((k // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t(), tile = tile)
    q = layer.B

    s = layer.s
    return q, s