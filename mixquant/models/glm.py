from .base import BaseForCausalLM
from typing import Dict
 


from mixquant.modules.fused.mlp import  MixChatGLMMLP
from mixquant.utils.utils import set_module_name
import torch
from typing import Optional, Tuple, Union, List
from torch import nn
from torch.nn import functional as F

class ChatGLMMixQForCausalLM(BaseForCausalLM):
    layer_type = "ChatGLMDecoderLayer"

    @staticmethod
    def fuse_layers(model , quant_config: Dict,  mix, cache):
        fuser = ChatGLMFuser(model)


        fuser.fuse_mlp(mix, cache)

    @staticmethod
    def get_model_layers(model ):
        return model.transformer.encoder.layers
    
    @staticmethod
    def move_embed(model , device):
        model.transformer.embedding.word_embeddings = model.transformer.embedding.word_embeddings.to(device)
    



class ChatGLMFuser:
    def __init__(self, model ):
        self.model = model

        self.attention_modules = [
            (name, module) for name, module in self.model.named_modules()
            if  "Attention" in str(module.__class__)
        ]
        self.mlp_modules = [
            (name, module) for name, module in self.model.named_modules()
            if    "MLP" in str(module.__class__)
        ]
   
    def fuse_mlp(self,mix, MixGemmCache = None):
        for name, module in self.mlp_modules:
            if  mix:
                assert MixGemmCache is not None
                mlp = MixChatGLMMLP(module.dense_h_to_4h, module.dense_4h_to_h, MixGemmCache)
            set_module_name(self.model, name, mlp)