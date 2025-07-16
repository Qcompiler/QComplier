export PYTHONPATH=/home/chenyidong/QComplier/src
CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
CMD=" srun -N 1 --gres=gpu:4090:1 --pty python"
set -x

base=/home/dataset/mixqdata
model=( Llama-2-7b )
model=( Aquila2-7b )
model=( Baichuan2-7b )
model=( Llama3-Chinese_v2)
model=( "Qwen2-7B-Instruct" )
model=( Llama-3.2-3B-Instruct )
$CMD examples/smooth_quant_get_act.py  --model-name  ${base}/${model}  \
        --output-path ${PYTHONPATH}/act_scales/${model}.pt \
         --dataset-path /home/dataset/mixqdata/val.jsonl.zst 

