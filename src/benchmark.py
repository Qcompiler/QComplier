
import os
os.environ["WORLD_SIZE"] = "1"
import time
import torch
import argparse
import numpy as np
import pandas as pd
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from torch.cuda import OutOfMemoryError

def warmup(model):
    warm_up = torch.randn((4096,4096)).to(next(model.parameters()).device)
    torch.mm(warm_up,warm_up)


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM, modeling_utils
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    return model
    
def generate(model, input_ids, n_generate):
    context_time = 0
    generate_time = []

    with torch.inference_mode():
        for i in range(n_generate):
            torch.cuda.synchronize()
            start = time.time()

            if i == 0:
                # prefill context
                inputs = torch.as_tensor(input_ids, device=next(model.parameters()).device)
            else:
                # decode tokens
                inputs = torch.as_tensor(token, device=next(model.parameters()).device)
            
            out = model.generate(inputs, max_length = 128)
            print(out)
            print(out[0].shape)
            exit(0)

            torch.cuda.synchronize()
            token = out[0][:, -1].max(1)[1].unsqueeze(1)

            if i == 0:
                context_time += time.time() - start
            else:
                generate_time.append(time.time() - start)
    
    return context_time, generate_time

def run_round(model_path, quant_file, n_generate, input_ids, batch_size, safetensors):
    print(f" -- Loading model...")
    model = AutoAWQForCausalLM.from_quantized(
        model_path, quant_file, fuse_layers=True,
        max_new_tokens=n_generate, batch_size=batch_size,
        safetensors=safetensors
    )

    print(f" -- Warming up...")
    warmup(model)

    print(f" -- Generating {n_generate} tokens, {input_ids.shape[1]} in context...")
    
    try:
        context_time, generate_time = generate(model, input_ids, n_generate)
        successful_generate = True
    except RuntimeError as ex:
        if 'cuda out of memory' in str(ex).lower():
            successful_generate = False
        else:
            raise RuntimeError(ex)
    
    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    if successful_generate:
        # number of tokens in context / time for processing context * batch size
        prefill_tokens_per_second = input_ids.shape[1] / context_time * batch_size
        # 1 second / median time per token in seconds * batch size
        decode_tokens_per_second = 1 / np.median(generate_time) * batch_size

        print(f" ** Speed (Prefill): {prefill_tokens_per_second:.2f} tokens/second")
        print(f" ** Speed (Decode): {decode_tokens_per_second:.2f} tokens/second")
        print(f" ** Max Memory (VRAM): {memory_used:.2f} GB ({memory_pct:.2f}%)")
    else:
        prefill_tokens_per_second = 'OOM'
        decode_tokens_per_second = 'OOM'

    return {
        "Batch Size": batch_size,
        "Prefill Length": input_ids.shape[1],
        "Decode Length": n_generate,
        "Prefill tokens/s": prefill_tokens_per_second,
        "Decode tokens/s": decode_tokens_per_second,
        "Memory (VRAM)": f"{memory_used:.2f} GB ({memory_pct:.2f}%)"
    }, model.quant_config["version"]

def main(args):
    rounds = [
        {"context": 32, "n_generate": 32},
        {"context": 64, "n_generate": 64},
        {"context": 128, "n_generate": 128},
        {"context": 256, "n_generate": 256},
        {"context": 512, "n_generate": 512},
        {"context": 1024, "n_generate": 1024},
        {"context": 2048, "n_generate": 2048},
    ]

    all_stats = []
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    for settings in rounds:
        input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, settings["context"])).cuda()

        stats, model_version = run_round(
            args.model_path,
            args.quant_file,
            settings["n_generate"],
            input_ids,
            args.batch_size,
            args.safetensors
        )
        
        all_stats.append(stats)

        if stats["Prefill tokens/s"] == 'OOM':
            break
    
    df = pd.DataFrame(all_stats)
    print('GPU:', torch.cuda.get_device_name())
    print('Model:', args.model_path)
    print('Version:', model_version)
    print(df.to_markdown(index=False))
    df.to_csv(args.quant_file.split("/")[-1]+".csv"+str(args.batch_size))

if __name__ == "__main__":

    """
    python examples/benchmark.py --model_path /mnt/data/zhongrx/Llama-2-7b-hf --quant_file /mnt/data/chenyd/Llama-2-7b-awq 

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="casperhansen/vicuna-7b-v1.5-awq", help="path to the model")
    parser.add_argument("--quant_file", type=str, default="awq_model_w4_g128.pt", help="weights filename")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for cache and generation")
    parser.add_argument("--safetensors", default=False, action="store_true", help="Use for enabling safetensors")
    args = parser.parse_args()

    main(args)