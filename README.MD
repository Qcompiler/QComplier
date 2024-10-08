# Quantize the LLMs with Mixed-precsion 


* Environment:
  * CUDA: 11.8, 12.1
  * Python: 3.10, 3.11


# Setup

build the kernel by
```
cd EETQ
python setup.py install

cd quantkernel
python setup.py install
```

# Quantize the 8-bit and 4-bit mixed-precision LLMs

## generate the act_scales by smoothquant

python examples/smooth_quant_get_act.py  --model-name /dev/shm/tmp/${model}  \
        --output-path ${base}/act_scales/${model}.pt  --dataset-path /home/cyd/val.jsonl.zst 