import torch
import os
from safetensors import safe_open
base = "/home/cyd/dataset/quant4"
source_awq =  os.path.join(base,"Qwen2-7B-Instruct-AWQ")
source_mix =  os.path.join(base,"Qwen2-7B-Instruct")
target_mix =  os.path.join(base+"mix","Qwen2-7B-Instruct")
map = { 'pytorch_model-00001-of-00002.bin' : "model-00001-of-00002.safetensors", 
        'pytorch_model-00002-of-00002.bin' : "model-00002-of-00002.safetensors", 

        }
try:
    os.mkdir(target_mix)
except:
    pass
files = os.listdir(source_mix)

for f in files:
    if "model-" in f:
        print(os.path.join(source_mix,f))
        f_awq = map[f]

      
        mix = torch.load(os.path.join(source_mix,f),  "cpu")
        awq =  safe_open(os.path.join(source_awq,f_awq), framework="pt", device="cpu") 
        #print(mix)
        to_delete = []
        for k in mix.keys():
            if "q_weight" in k or "q_scale_col"  in k or ".weight" in k  or "weight_cache" in k or "scale_col" in k:
                to_delete.append(k)

        for key in to_delete:
            del mix[key]

        print(mix.keys())
        for k in awq.keys():
            mix[k] = awq.get_tensor(k)
        torch.save(mix, os.path.join(target_mix,f))


    
# 拷贝其他的文件

os.system("cp -r " + source_mix + "/*.json " + target_mix )
os.system("cp -r " + source_mix + "/*.txt " + target_mix )


# 修改  pytorch_model.bin.index.json

import json
with open(os.path.join(source_mix, "pytorch_model.bin.index.json"), 'r') as fcc_file:
    fcc_data = json.load(fcc_file)

weight = fcc_data['weight_map']

for key in to_delete:
    del weight[key]

new_key = {}
for key in weight.keys():
    v = weight[key]
    if "ind" in key:
        
        key1 = key.replace("ind", "qweight")
        key2 = key.replace("ind", "qzeros")
        key3 = key.replace("ind", "scales")
        new_key[key1] = v
        new_key[key2] = v
        new_key[key3] = v

fcc_data['weight_map'] = new_key

with open(os.path.join(target_mix, "pytorch_model.bin.index.json"), 'w') as f:
    json.dump(fcc_data, f)




# for f in files:
#     if "model-" in f:
#         print(os.path.join(source_mix,f))
#         f_awq = map[f]

      
#         mix = torch.load(os.path.join(source_mix,f),  "cpu")
#         awq =  safe_open(os.path.join(source_awq,f_awq), framework="pt", device="cpu") 
#         #print(mix)
#         to_delete = []
#         for k in mix.keys():
#             if "q_weight" in k or "q_scale_col"  in k or ".weight" in k  or "weight_cache" in k or "scale_col" in k or "ind" in k:
#                 to_delete.append(k)

#         for key in to_delete:
#             del mix[key]

#         print(mix.keys())
#         for k in awq.keys():
#             mix[k] = awq.get_tensor(k)
#         torch.save(mix, os.path.join(target_mix,f))

