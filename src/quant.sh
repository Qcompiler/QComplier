
 
CMD=" python "
#CMD=" python" 
 
set -x

# model=65      
# CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
# python examples/basic_quant_mix.py  \
# --model_path /home/dataset/llama-2/checkpoint/Llama-${model}b \
# --quant_file /home/dataset/llama-2/checkpoint/quant/Llama-${model}b


models=(  "Baichuan2-7b"  "Baichuan2-13b" "Aquila2-7b" "Llama-2-7b"  "Mistral-7b" )
models=(  "Llama-2-7b" "vicuna-7b"  "Aquila2-7b" "falcon-7b" "Baichuan2-7b")

models=(  "Llama-2-1b")
# models=(  "Baichuan2-7b")
# models=( "vicuna-7b")
# models=( "Llama3-Chinese_v2")
models=( "Qwen2-7B-Instruct" )
path=/home/cyd/dataset
for bit in   4  8 
  do
    for model in "${models[@]}"
            do
            echo ${model}
            ${CMD} \
              examples/basic_quant_mix.py  \
            --model_path ${path}/${model} \
            --quant_file ${path}/quant${bit}/${model} --w_bit ${bit}
    done
done


