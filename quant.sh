export PYTHONPATH=/home/chenyidong/QComplier/src
CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
CMD=" srun -N 1 --gres=gpu:H100:1 python"
set -x

export PYTHONPATH=/home/chenyidong/QComplier
models=( $1 )
path=$2
for bit in   4
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


