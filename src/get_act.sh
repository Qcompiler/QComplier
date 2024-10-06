
CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
CMD="python"
set -x

base=/home/cyd/chitu-vllm/QComplier/src
model=( Llama-2-7b )
model=( Aquila2-7b )
model=( Baichuan2-7b )
model=( Llama3-Chinese_v2)
model=( "Qwen2-7B-Instruct" )
#model=( "chatglm/chatglm2-6b" )
$CMD examples/smooth_quant_get_act.py  --model-name /dev/shm/tmp/${model}  \
        --output-path ${base}/act_scales/${model}.pt  --dataset-path /home/cyd/val.jsonl.zst 

