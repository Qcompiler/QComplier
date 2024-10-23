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

 Download the ``` val.jsonl.zst``` from

https://huggingface.co/datasets/mit-han-lab/pile-val-backup/blob/main/val.jsonl.zst

 generate the act_scales by smoothquant

```
export PYTHONPATH=$( pwd )
export model=/home/chenyidong/data/mixqdata/Llama-2-7b
python examples/smooth_quant_get_act.py  --model-name ${model}  \
        --output-path ${PYTHONPATH}/act_scales/Llama-2-7b.pt \
         --dataset-path /home/chenyidong/data/mixqdata/val.jsonl.zst 
```

## Quantize the Model


```
export PYTHONPATH=$( pwd )
export bit=8
python examples/basic_quant_mix.py  \
            --model_path ${model} \
            --quant_file /home/chenyidong/data/mixqdata/quant${bit}/Llama-2-7b --w_bit ${bit}
```



