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

## Download the ``` val.jsonl.zst``` from

https://huggingface.co/datasets/mit-han-lab/pile-val-backup/blob/main/val.jsonl.zst

## generate the act_scales by smoothquant

python examples/smooth_quant_get_act.py  --model-name /dev/shm/tmp/${model}  \
        --output-path ${base}/act_scales/${model}.pt  --dataset-path /home/cyd/val.jsonl.zst 

## Quantize the Model

```
bash quant.sh
```
or
```
export bit=8
python examples/basic_quant_mix.py  \
            --model_path ${path}/${model} \
            --quant_file ${path}/quant${bit}/${model} --w_bit ${bit}
```
