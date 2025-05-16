"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import logging
import random
import time

import numpy as np
import torch
from bayes_opt import BayesianOptimization, UtilityFunction
from fastchat.utils import str_to_torch_dtype
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from evaluation.eval import reorg_answer_file, run_eval
from model.myswift.kv_cache import initialize_past_key_values
from model.myswift.modeling_llama import LlamaForCausalLM
from model.myswift.utils import (
    evaluate_posterior,
    generate_candidates,
    generate_swift_buffers,
    get_cache_configuration,
    get_choices_list,
    get_choices_list_notree,
    initialize_swift,
    prepare_logits_processor,
    reset_swift_mode,
    set_logger,
    swift_draft,
    swift_optimization,
    tree_decoding,
    update_inference_inputs,
)
from model.pld.pld import greedy_search_pld

first_acc_rates = []
def swift_forward(inputs, model, tokenizer, max_new_tokens, statistics=None, optimizer=None, utility=None,
                  logits_processor=None, max_steps=512):
    input_ids = inputs.input_ids.cuda()
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    accept_length_list = []

    # Initialize the past key and value states
    (
        past_key_values,
        past_key_values_data,
        current_length_data,
    ) = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_swift_mode(model)
    start_prefill = time.time()
    swift_logits, sample_token, top1_prob = initialize_swift(input_ids, model, max_new_tokens,
                                                             past_key_values, past_key_values_data,
                                                             current_length_data, logits_processor=logits_processor)

    logging.info(f"Prefill time: {time.time() - start_prefill:.4f}s")
    # Clone the prefilled past key and value states for swift optimization
    input_past_key_values_data = []
    for i in range(len(past_key_values_data)):
        input_past_key_values_data.append(past_key_values_data[i].clone())
    input_current_length_data = current_length_data.clone()

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    
    timings = {
        "total_step": [],
        # "dp_optim": [],
        # "draft_clone_kv": [],
        "draft_loop": [],
        "avg_draft_time": [],
        "verify": [],
        "accept_update": [],
        "misc_overhead": [],
    }
    step_end_time = time.time()  # Initialize before loop

    for idx in range(max_steps):
        start_step = time.time()
        timings["misc_overhead"].append(start_step - step_end_time)
        
        # drafted tokens + 1 bonus verified token
        draft_token_num += len(top1_prob)
        # Initialize the swift buffer
        swift_choices = eval(f"{get_choices_list_notree(top1_prob, logits_processor=logits_processor)}")
        # logging.info(f"Swift choices: {swift_choices}")
        swift_buffers = generate_swift_buffers(swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device)
        model.swift_buffers = swift_buffers
        model.swift_choices = swift_choices
        model.model.swift_mask = swift_buffers["swift_attn_mask"]

        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            swift_logits,
            swift_buffers["tree_indices"],
            swift_buffers["retrieve_indices"],
            sample_token,
            logits_processor
        )

        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            swift_buffers["swift_position_ids"],
            input_ids,
            swift_buffers["retrieve_indices"],
        )
        
        verify_end_time = time.time()
        timings["verify"].append(verify_end_time - start_step)


        best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor, cart_candidates_prob, swift_logits[2],
                swift_buffers["p_indices"], tree_candidates, swift_buffers["b_indices"]
            )

        if accept_length == 0:
            first_acc_rates.append(0)
        else:
            first_acc_rates.append(1)

        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            swift_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p
        )
        accept_update_end_time = time.time()
        timings["accept_update"].append(accept_update_end_time - verify_end_time)


        # layer set optimization
        if (new_token_num > (statistics["context_window"] + 1) and statistics["optimization"]
                and idx % statistics["opt_interval"] == 0):
            swift_optimization(
                model,
                input_ids[:, input_len:],
                input_past_key_values_data,
                input_current_length_data,
                new_token_num,
                statistics,
                optimizer=optimizer,
                utility=utility)



        # swift drafting
        swift_logits, top1_prob = swift_draft(
            model,
            input_ids=sample_token,
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
        )
        
        draft_loop_end_time = time.time()
        draft_time = draft_loop_end_time - accept_update_end_time
        timings["draft_loop"].append(draft_time)
        timings["avg_draft_time"].append(
            draft_time / len(top1_prob)
        )
        
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break
        
        step_end_time = time.time()
        timings["total_step"].append(step_end_time - start_step)
        
        
    # logging.info("token acceptance rate: {}".format(total_acc_num / draft_token_num))
    print("Total acceptance rate: {}".format(total_acc_num / draft_token_num))
    
    # --- Print Timings ---
    logging.info("--- Performance Timings (Average per Step) ---")
    for key, values in timings.items():
        if values:
             avg_time = np.mean(values)
             logging.info(f"{key}: {avg_time:.4f}s")
        else:
             logging.info(f"{key}: N/A (not run or no steps)")
    logging.info(f"Average Acceptance Length: {np.mean(accept_length_list) - 1:.4f}")
    logging.info(
        f"Finished. Total steps: {idx}, Total generated: {new_token_num}"
    )

    return input_ids, new_token_num, idx + 1, accept_length_list #, draft_token_num

def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1034,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for swift sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
    )
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.45,
        help="The skipped layer ratio of swift.",
    )
    parser.add_argument(
        "--opt-interval",
        type=int,
        default=1,
        help="The interval of swift optimization.",
    )
    parser.add_argument(
        "--bayes-interval",
        type=int,
        default=25,
        help="The interval of bayesian optimization.",
    )
    parser.add_argument(
        "--max-opt-iter",
        type=int,
        default=1000,
        help="The maximum layer set optimization iteration.",
    )
    parser.add_argument(
        "--max-tolerance-iter",
        type=int,
        default=300,
        help="The maximum tolerance of layer set search iteration.",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.95,
        help="The early stop threshold of layer set search.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=32,
        help="The context window of swift.",
    )
    parser.add_argument(
        "--optimization",
        action="store_true",
        default=False,
        help="Layer set optimization.",
    )
    parser.add_argument(
        "--bayes",
        action="store_true",
        default=False,
        help="Bayes Optimization of Layer set.",
    )
    parser.add_argument(
        "--cache-hit",
        action="store_true",
        default=False,
        help="Whether to use cached SWIFT configuration.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="The sampling seed.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--no-tree",
        action="store_true",
        default=False,
        help="Whether to use tree decoding.",
    )

    args = parser.parse_args()

    args.model_name = (args.model_id + "-myswift-" + str(args.dtype)+ "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed) + "-max_new_tokens-" + str(args.max_new_tokens)+ "-opt_interval-" + str(args.opt_interval)
                       + "-bayes_interval-" + str(args.bayes_interval) + "-max_opt-" + str(args.max_opt_iter) + "-max_tolerance-" + str(args.max_tolerance_iter)
                       + "-max_score-" + str(args.max_score) + "-context_window-" + str(args.context_window) + "-skip_ratio-" + str(args.skip_ratio))
    answer_file = f"data/{args.bench_name}/model_answer/{args.model_name}.jsonl"
    set_logger()

    print(f"Output to {answer_file}")

    question_file = f"data/{args.bench_name}/question.jsonl"
    
    if args.answer_file:
        answer_file = args.answer_file

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature, top_p=args.top_p)
    else:
        logits_processor = None

    if args.cache_hit:
        # Load the cached layer set configuration
        args.optimization, args.bayes=False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(model_name=args.model_id,
                                                                                  bench_name=args.bench_name)
    else:
        # Unified layer set initialization
        _attn_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)  # keep the first and last layer
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)
    
    model.greedy_search_pld = greedy_search_pld.__get__(model, type(model))

    seed_everything(args.seed)
    # Bayes Optimization Settings
    pbounds = {f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)} # keep the first and last layer
    optimizer = BayesianOptimization(f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True)
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {"origin_score": 0, "opt_iter": 0, "tolerance_iter": 0,
                  "skip_ratio": args.skip_ratio, "acceptance_rate_list": [], "opt_interval": args.opt_interval,
                  "bayes_interval": args.bayes_interval, "max_opt_iter": args.max_opt_iter,
                  "max_tolerance_iter": args.max_tolerance_iter, "max_score": args.max_score,
                  "context_window": args.context_window, "optimization": args.optimization, "bayes": args.bayes}

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        # data_num=args.data_num,
        # seed=args.seed,
        optimizer=optimizer,
        utility=utility,
        statistics=statistics,
        logits_processor=logits_processor,
    )
    
    print("First acceptance rate: ", np.mean(first_acc_rates))
    
    reorg_answer_file(answer_file)
