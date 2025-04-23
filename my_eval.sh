MODEL_NAME=vicuna-7b-v1.3
Vicuna_PATH=/gemini/user/shared/models/$MODEL_NAME

TEMP=0.0
GPU_DEVICES=0

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --use-csd-mgram
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --use-csd-mgram --fallback "data"

python evaluation/speed.py \
    --base-path data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
    --file-path data/spec_bench/model_answer/vicuna-7b-v1.3-pld-float16_csd.jsonl \
    --tokenizer-path $Vicuna_PATH

python evaluation/speed.py \
    --base-path data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
    --file-path data/spec_bench/model_answer/vicuna-7b-v1.3-pld-float16_csd_data.jsonl \
    --tokenizer-path $Vicuna_PATH