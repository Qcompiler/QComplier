import torch
from typing import Dict, List, Optional
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from mixquant.utils.utils import clear_memory
from mixquant.utils.calib_data import get_calib_dataset
from mixquant.modules.linear import MixLinear_GEMM

from mixquant.utils.module import get_named_linears, set_op_by_name, weight_only_map, eightbit_only_name

from EETQ import quant_weights
class MixQuantizer:
    def __init__(self, f16_model, model, tokenizer, w_bit, group_size, version) -> None:
        self.f16_model = f16_model
        self.model = model
        self.tokenizer = tokenizer

        self.group_size = group_size
        self.version = version
        self.w_bit = w_bit

        self.modules, self.module_kwargs= self.init_quant()
    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.f16_model.get_model_layers(self.model)


        inps = []
        layer_kwargs = {}

        modules[0] = modules[0].cuda()
        self.f16_model.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hijacked_inputs, **kwargs):
                inps.append(hijacked_inputs)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        modules[0] = Catcher(modules[0])

        modules[0] = modules[0].module  # restore

        modules[0] = modules[0].cpu()
        self.f16_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if "attention_mask" in layer_kwargs.keys():
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs
    

    def quantize(self,weight_only = False):
        for i in tqdm(range(len(self.modules)), desc="Mix quant"):

            self.modules[i] = self.modules[i].cuda()
            named_linears = get_named_linears(self.modules[i])

            clear_memory()

            # Quantize weights
            arch = torch.cuda.get_device_capability()
            arch = arch[0] * 10 + arch[1]

            print("arch is ")
            print(arch)

            self._apply_quant(self.modules[i], named_linears, weight_only, layer = i, arch = arch)

            clear_memory()
 


    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear], weight_only_, layer, arch):

        
        if isinstance(self.model.config.architectures,list):
            name = self.model.config.architectures[0]
        else:
            name = self.model.config.architectures
        weight_only_name = weight_only_map[ name ]
        print(named_linears.items())
        merged = False
        up_merged = False

        print("model name is")
        print(name)
        if "GLM" in name:
            merged = True
            up_merged = True
        if "Baichuan" in name:
            merged = True
            up_merged = False
        
        if not merged:
            q, k, v =   named_linears['self_attn.q_proj'].weight.data, \
                        named_linears['self_attn.k_proj'].weight.data, named_linears['self_attn.v_proj'].weight.data, 
            qshape =  q.shape
            kshape =  k.shape
            vshape =  v.shape
        
        if not up_merged:
            gate = named_linears['mlp.gate_proj'].weight.data
            up = named_linears['mlp.up_proj'].weight.data
            upshape = up.shape
            
            gateshape = gate.shape
        if not merged:
            merged_qvk = torch.nn.Linear(q.shape[1], qshape[0] + kshape[0] + vshape[0], bias = False  )
            merged_qvk.weight.data = torch.cat( [q,  k , v], dim = 0)

            test_error = False
            
                
            int8_weight_cpu = torch.t(merged_qvk.weight.data).contiguous().cpu()
            qweight, q_scale_col = quant_weights(int8_weight_cpu, torch.int8, False)

            # test for error
        print("current layer is ",layer)
        
        if not up_merged:
            assert up.shape[1] == gate.shape[1]
            merged_up_gate = torch.nn.Linear(up.shape[1], upshape[0] + gateshape[0], bias = False  )
            merged_up_gate.weight.data = torch.cat( [gate,  up], dim = 0)

            
            int8_weight_cpu = torch.t(merged_up_gate.weight.data).contiguous().cpu()
            gate_up_qweight, gate_up_q_scale_col = quant_weights(int8_weight_cpu, torch.int8, False)


        #print(gate_up_qweight)
        # than we do not need to use the qkv or update!!!
        for name, linear_layer in named_linears.items():


            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            if self.version == 'MIX':
                
                q_linear_module = MixLinear_GEMM

            else:
                raise NotImplementedError
            
            # for same small blocks we do not need the mixquant, we only use the weight only quant

            weight_only = False

            for key in weight_only_name:
                if key in  name:
                    weight_only = True
                    break
                
            w_bit = self.w_bit
            if w_bit == 4:
                for key in eightbit_only_name:
                    if key in  name:
                        w_bit = 8
                        weight_only = False 


            relative_path = "act_scales/%s.pt"%(self.model.config._name_or_path.split("/")[-1])

            
            act_scales = torch.load(relative_path)


            if 'opt' in self.model.config._name_or_path.split("/")[-1]:
                layer_scales = act_scales['model.decoder.layers.{}.{}'.format(layer, name)]

            elif 'falcon' in self.model.config._name_or_path.split("/")[-1]:
                layer_scales = act_scales['transformer.h.{}.{}'.format(layer, name)]    
            elif 'Baichuan' in self.model.config._name_or_path.split("/")[-1]:

                layer_scales = act_scales['model.layers.{}.{}'.format(layer, name)]
            elif "glm" in self.model.config._name_or_path.split("/")[-1]:
                layer_scales = act_scales['transformer.encoder.layers.{}.{}'.format(layer, name)]
            else:
                layer_scales = act_scales['model.layers.{}.{}'.format(layer, name)]

            fp_features_num = 128

 

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                weight_only = weight_only,
                init_only=False,
                bit = w_bit,
                layer_scales = layer_scales,
                fp_features_num = fp_features_num,
                arch = arch
            )

   

            linear_layer.cpu()

            print(name)
            if w_bit == 8:
                if "q_proj" in name:
                    print("replace q_proj")
                    assert q_linear.q_weight.shape == qweight[:, 0 : qshape[0]].shape
                    q_linear.q_weight =  qweight[ :,  0 :    qshape[0] ].contiguous() 
                    q_linear.q_scale_col.copy_(q_scale_col[0 : qshape[0]])

                    
                if "k_proj" in name:
                    print("replace k_proj")
                    assert q_linear.q_weight.shape == qweight[: , qshape[0] : qshape[0] + kshape[0]].shape
                    q_linear.q_weight.copy_( qweight[:,  qshape[0] : (qshape[0] + kshape[0]   )  ].contiguous() )
                    q_linear.q_scale_col.copy_( q_scale_col[qshape[0] : qshape[0] + kshape[0]])

                if "v_proj" in name:
                    print("replace v_proj")
                    assert q_linear.q_weight.shape == qweight[:, qshape[0] + kshape[0] : qshape[0] + kshape[0] + vshape[0]].shape
                    q_linear.q_weight.copy_(  qweight[:,   (qshape[0] + kshape[0])   :   (qshape[0] + kshape[0] + vshape[0]) ].contiguous() )
                    q_linear.q_scale_col.copy_(q_scale_col[qshape[0] + kshape[0] : qshape[0] + kshape[0] + vshape[0]])

                
                if "gate_proj" in name:
                    print("replace gate_proj")
                    assert q_linear.q_weight.shape == gate_up_qweight[:, 0 : gateshape[0]].shape    
                    q_linear.q_weight.copy_(gate_up_qweight[:,  0 :    gateshape[0]].contiguous()   )
                    q_linear.q_scale_col.copy_( gate_up_q_scale_col[0 : gateshape[0]])



                if "up_proj" in name:
                    print("replace up_proj")
                    assert q_linear.q_weight.shape == gate_up_qweight[ :, gateshape[0] : upshape[0] + gateshape[0]].shape   
                    q_linear.q_weight.copy_( gate_up_qweight[ :,  gateshape[0] :  (upshape[0] + gateshape[0])].contiguous()   )
                    q_linear.q_scale_col.copy_( gate_up_q_scale_col[gateshape[0] : upshape[0] + gateshape[0]])
    

            set_op_by_name(module, name, q_linear)
            clear_memory()
        
            

    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        # get repeated count
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        # dequantize
        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w
    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0
        self.zero_point = True
        # zero point quantization
        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros