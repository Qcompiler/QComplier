export PYTHONPATH=/home/chenyidong/QComplier/src
CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
CMD="   python"
set -x

export PYTHONPATH=/home/cyd/QComplier
models=( $1 )
path=$2
for bit in    8 
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


