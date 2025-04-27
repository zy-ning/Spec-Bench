MODEL_NAME=vicuna-7b-v1.3
Vicuna_PATH=/data/nzy/models/$MODEL_NAME

TEMP=0.0
GPU_DEVICES=4

DATA_NUM=100
SEED=2024
GPU_DEVICES=0
MAX_NEW_TOKENS=1024

# SWIFT Hyperparameters
OPT_INTERVAL=1
BAYES_INTERVAL=25
MAX_OPT_ITER=1000
MAX_TOLERANCE_ITER=300
MAX_SCORE=0.93
CONTEXT_WINDOW=50
SKIP_RATIO=0.45

bench_NAME="spec_bench"
torch_dtype="float16" # ["float32", "float64", "float16", "bfloat16"]

# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_baseline --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-vanilla-${torch_dtype}-temp-${TEMP} --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --use-csd-mgram
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_pld --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-pld-${torch_dtype} --bench-name $bench_NAME --dtype $torch_dtype --use-csd-mgram --fallback "model"
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_recycling --model-path $Vicuna_PATH --model-id ${MODEL_NAME}-recycling --bench-name $bench_NAME --temperature $TEMP --dtype $torch_dtype
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation.inference_swift --model-path $Vicuna_PATH --model-id ${MODEL_NAME} \
  --temperature $TEMP --dtype $torch_dtype --bench-name $bench_NAME --data-num ${DATA_NUM} --max-new-tokens ${MAX_NEW_TOKENS} \
  --seed $SEED --context-window ${CONTEXT_WINDOW} --opt-interval ${OPT_INTERVAL} --bayes-interval ${BAYES_INTERVAL} --max-opt-iter ${MAX_OPT_ITER} \
  --max-tolerance-iter ${MAX_TOLERANCE_ITER} --max-score ${MAX_SCORE} --skip-ratio ${SKIP_RATIO} --optimization --bayes # --cache-hit


# python evaluation/speed.py \
#     --base-path data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
#     --file-path data/spec_bench/model_answer/vicuna-7b-v1.3-pld-float16.jsonl \
#     --tokenizer-path $Vicuna_PATH

# python evaluation/speed.py \
#     --base-path data/spec_bench/model_answer/vicuna-7b-v1.3-vanilla-float16-temp-0.0.jsonl \
#     --file-path data/spec_bench/model_answer/vicuna-7b-v1.3-recycling.jsonl \
#     --tokenizer-path $Vicuna_PATH
