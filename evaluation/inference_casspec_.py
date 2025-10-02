import argparse
import logging
import random
import time  # For timing checks

import numpy as np
import torch

# from distribution import Distribution
from fastchat.utils import str_to_torch_dtype
from transformers import AutoTokenizer

# Assuming these are correctly adapted or available:
from evaluation.eval import reorg_answer_file, run_eval
from model.casspec.kv_cache import (
    clone_past_key_values,
    initialize_past_key_values,
)
from model.casspec.modeling_llama import (
    LlamaForCausalLM,  # Ensure this class is adapted as per previous steps
)

# Import CLASP utils
from model.casspec.utils import (  # Ensure this points to your utils
    CLaSp_Skip_Layer_Strategy_SeqParallel,  # The core DP algorithm
    cassepc_draft,
    evaluate_posterior,
    generate_candidates,
    generate_swift_buffers,
    get_choices_list,
    prepare_logits_processor,
    # sample,
    set_logger,
    tree_decoding,
    update_inference_inputs,
)

# Import pld
# from model.pld.pld import find_candidate_pred_tokens

# TEAL
# from kernels.sparse_gemv import SparseGEMV, SparseQKVGEMV, DenseGEMV


@torch.no_grad()
def casspec_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    statistics=None,
    logits_processor=None,
    max_steps=512,
    args=None,
    # see_token=True,
    see_token=False,
):
    global steps_since_last_optim, current_draft_skip_mask
    input_ids = inputs.input_ids.cuda()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only support batch size 1 for now!!"
    # --- Initialization ---
    new_token_num = 0
    total_steps = 0
    accept_length_list = []
    L = model.config.num_hidden_layers
    M = int(L * args.skip_ratio)  # Number of layers to skip (M for CLaSp DP)
    optim_interval_steps = args.opt_interval
    K = args.draft_length_K
    DET = args.draft_exit_threshold

    past_key_values, past_key_values_data, current_length_data = (
        initialize_past_key_values(model.model)
    )  # Pass base model
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len

    # --- Initial Prefill ---
    start_prefill = time.time()
    prefill_outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        output_hidden_states=True,
    )
    prefill_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1)
    if see_token:
        logging.info(
            f"\nPrefill token: {prefill_token.item()}, str of token: {tokenizer.convert_ids_to_tokens(prefill_token.item())}"
        )
    sample_token = prefill_token.unsqueeze(1)
    # input_ids = torch.cat([input_ids, prefill_token.unsqueeze(1)], dim=1)
    new_token_num += 1
    logging.info(f"Prefill time: {time.time() - start_prefill:.4f}s")

    max_cache_len = getattr(model.config, "max_position_embeddings", 2048)
    if max_new_tokens > max_cache_len - input_len:
        logging.info(
            f"Warning: max_new_tokens ({max_new_tokens}) exceeds max cache length ({max_cache_len - input_len})."
        )
        max_new_tokens = max_cache_len - input_len - 1
    # dp_statistics = {
    #     "accepted_len": [],
    #     "draft_len": [],
    # }
    last_accepted_token_hidden_states = None
    if prefill_outputs.hidden_states:
        last_accepted_token_hidden_states = [
            s[:, -1, :].squeeze(0) for s in prefill_outputs.hidden_states
        ]

    # --- Generation Loop ---
    timings = {
        "total_step": [],
        "dp_optim": [],
        "draft_loop": [],
        "avg_draft_time": [],
        "verify": [],
        "accept_update": [],
        "misc_overhead": [],
    }
    step_end_time = time.time()  # Initialize before loop

    while new_token_num < max_new_tokens and total_steps < max_steps:
        total_steps += 1
        start_step = time.time()
        timings["misc_overhead"].append(start_step - step_end_time)

        # --- 1. Layer Optimization (Run periodically) ---
        dp_start_time = time.time()
        run_dp = False
        if last_accepted_token_hidden_states is not None and (
            current_draft_skip_mask is None
            or steps_since_last_optim >= optim_interval_steps
        ):
            start_dp = time.time()
            # logging.info(
            #     f"CLaSp DP: last statistics: avg accepted len: {np.mean(dp_statistics['accepted_len']) - 1}, "
            #     f"avg draft len: {np.mean(dp_statistics['draft_len']) - 1}, "
            #     f"avg accepted rate: {np.mean([0 if al == 1 else 1 for al in dp_statistics['accepted_len']])}"
            # )
            opt_past_key_values_data_list = [d.clone() for d in past_key_values_data]
            opt_current_length_data = (
                current_length_data.clone()
            )  # This tracks lengths for the opt cache
            # Create new KVCache objects pointing to the *cloned* data
            opt_kv_cache_list = clone_past_key_values(
                model, opt_past_key_values_data_list, opt_current_length_data
            )
            # logging.info(
            #     f"Hidden states shape: {last_accepted_token_hidden_states[0].shape}"
            # )
            # logging.info(f"Running CLaSp DP Optimization at step {total_steps}...")
            current_draft_skip_mask = CLaSp_Skip_Layer_Strategy_SeqParallel(
                L=L,
                M=M,
                hidden_states_H=last_accepted_token_hidden_states,
                model_layers=model.model.layers,
                past_key_values=opt_kv_cache_list,
                device=device,
            )
            # best_mask = [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
            # current_draft_skip_mask = torch.tensor(best_mask).bool().to(device)
            # logging.info("CLaSp DP Optimization: Layer to skip: ", current_draft_skip_mask)
            steps_since_last_optim = 0  # Reset counter
            logging.info(
                f"CLaSp DP time: {time.time() - start_dp:.4f}s. Skipped {torch.sum(current_draft_skip_mask)} layers.\n"
                f"Drafting with mask: {current_draft_skip_mask.int().cpu().tolist()}"
            )
            # dp_statistics = {
            #     "accepted_len": [],
            #     "draft_len": [],
            # }
        dp_end_time = time.time()
        if run_dp:
            timings["dp_optim"].append(dp_end_time - dp_start_time)

        # --- 2. Drafting (Autoregressive) ---
        start_draft = time.time()
        # first_draft_input_ids = input_ids[:, -1].unsqueeze(1)
        draft_tree_logits, top1_prob, is_eos = cassepc_draft(
            model=model,
            device=device,
            input_ids=torch.cat([input_ids, sample_token], dim=1),
            new_token_num=new_token_num,
            past_key_values_data=past_key_values_data,
            current_length_data=current_length_data,
            logits_processor=logits_processor,
            eos_token_id=tokenizer.eos_token_id,
            current_draft_skip_mask=current_draft_skip_mask,
            K=K,
            DET=DET,
        )
        draft_loop_end_time = time.time()
        timings["draft_loop"].append(draft_loop_end_time - start_draft)
        num_drafted = draft_tree_logits[0].shape[0]
        timings["avg_draft_time"].append(
            (draft_loop_end_time - start_draft) / num_drafted if num_drafted > 0 else 0
        )

        tree_choices = eval(
            f"{get_choices_list(top1_prob, logits_processor=logits_processor)}"
        )
        tree_buffers = generate_swift_buffers(tree_choices, device=device)
        model.model.swift_mask = tree_buffers["swift_attn_mask"]
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            draft_tree_logits,
            tree_buffers["tree_indices"],
            tree_buffers["retrieve_indices"],
            sample_token,
            logits_processor,
        )

        # --- 3. Verification (Parallel) ---
        current_seq_len = current_length_data[
            0
        ].item()  # Get current length BEFORE verify
        cur_input_len = input_ids.shape[1]
        if current_seq_len != cur_input_len:
            logging.info(
                f"Warning: Current length ({current_seq_len}) does not match input_ids length ({cur_input_len})."
            )
        start_verify = time.time()
        logits, verify_outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_buffers["swift_position_ids"],
            input_ids,
            tree_buffers["retrieve_indices"],
        )
        verify_hidden_states = verify_outputs.hidden_states
        verify_end_time = time.time()
        timings["verify"].append(verify_end_time - start_verify)

        # --- 4. Acceptance & Update ---
        start_accept = time.time()
        last_accepted_token_hidden_states = None
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits,
            candidates,
            logits_processor,
            cart_candidates_prob,
            draft_tree_logits[2],
            tree_buffers["p_indices"],
            tree_candidates,
            tree_buffers["b_indices"],
        )

        if verify_hidden_states:
            last_accepted_token_hidden_states = [
                s[0, tree_buffers["retrieve_indices"]][best_candidate, accept_length] for s in verify_hidden_states
            ]

        # --- Update Sequence and KV Cache Length ---
        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            tree_buffers["retrieve_indices"],
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p,
        )
        # input_ids = torch.cat([input_ids, sample_token], dim=1)

        accept_update_end_time = time.time()
        timings["accept_update"].append(accept_update_end_time - start_accept)

        timings["total_step"].append(accept_update_end_time - start_step)
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        steps_since_last_optim += 1
        # logging.info(
        #     f"Step {total_steps}: Accepted length: {accept_length_tree}, "
        #     f"Drafted length: {accept_length}\n"
        #     f"Sample token: {sample_token.item()}, str of token: {tokenizer.convert_ids_to_tokens(sample_token.item())}\n"
        #     f"Accepted Tokens{[tokenizer.convert_ids_to_tokens(t) for t in input_ids[0, -accept_length_tree:].tolist()]}\n"
        #     f"Generated token length: {new_token_num}, "
        # )
        # --- Check for EOS ---
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break

    # --- Final Output ---
    # (Output logic unchanged)
    avg_accept_len = np.mean(accept_length_list) if accept_length_list else 0

    # --- Print Timings ---
    # logging.info(f"\ntotal_steps: {total_steps}")
    logging.info("--- Performance Timings (Average per Step) ---")
    for key, values in timings.items():
        if values:
            avg_time = np.mean(values)
            logging.info(f"{key}: {avg_time:.4f}s")
        else:
            logging.info(f"{key}: N/A (not run or no steps)")
    logging.info(f"Average Acceptance Length: {(avg_accept_len - 1):.2f}")
    logging.info(
        f"Finished. Total steps: {total_steps}, Total generated: {new_token_num}"
    )

    return input_ids, new_token_num, total_steps, accept_length_list


def seed_everything(seed=64):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Main Execution Block ---
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
        default=1024,
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
        help="The temperature for clasp sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="The top-p for sampling.",
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
        "--question-end", type=int, help="A debug option. The end index of questions."
    )

    # --- for CLaSp ---
    parser.add_argument(
        "--skip-ratio",
        type=float,
        default=0.5,  # Example value, needs tuning
        # required=True,  # M is a key param for CLaSp DP
        help="The target number of layers to skip (M for CLaSp DP).",
    )
    parser.add_argument(
        "--opt-interval",
        type=int,
        default=1,  # Default: Optimize before every draft step
        help="The interval (in terms of accepted tokens) between CLaSp DP optimizations (Section 3.7).",
    )
    parser.add_argument(
        "--draft-exit-threshold",
        type=float,
        default=0.7,  # Example value, needs tuning
        help="Draft-Exiting Threshold (DET) based on draft model confidence (Section 5.3.3).",
    )
    parser.add_argument(
        "--draft-length-K",
        type=int,
        default=8,
        help="Maximum number of tokens to draft in each step (K).",
    )

    args = parser.parse_args()

    args.model_name = (
        args.model_id
        + "-casspec-"
        + str(args.dtype)
        + "-temp-"
        + str(args.temperature)
        # + "-topp-"
        # + str(args.top_p)
        # + "-seed-"
        # + str(args.seed)
        # + "-maxntok-"
        # + str(args.max_new_tokens)
        + f"-I{args.opt_interval}"
        + f"-K{args.draft_length_K}"
        + f"-DET{args.draft_exit_threshold}"
        + f"-skip{args.skip_ratio}"
    )  # Include CLaSp params
    answer_file = f"data/{args.bench_name}/model_answer/{args.model_name}.jsonl"
    set_logger()  # Assuming set_logger is in clasp_utils or imported

    print(f"Output to {answer_file}")
    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file

    # --- Model Loading ---
    print(f"Loading model: {args.model_path}")
    # Crucially, ensure output_hidden_states can be enabled if not default
    # config_kwargs = {"output_hidden_states": True} # Or modify config before loading
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
        # config=AutoConfig.from_pretrained(args.model_path, **config_kwargs) # If needed
    )
    model.eval()  # Ensure dropout etc. are off

    # Explicitly enable in config if needed AFTER loading
    model.config.output_hidden_states = True
    # Also for the base model if structure is nested e.g. model.model
    if hasattr(model, "model"):
        model.model.config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("Model and tokenizer loaded.")

    # --- Logits Processor ---
    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=args.temperature, top_p=args.top_p
        )
    else:
        logits_processor = None

    # --- CLaSp specific setup ---
    # No fixed layer set needed initially, it's dynamic.
    seed_everything(args.seed)  # Assuming seed_everything is available

    # No Bayes optimizer setup needed for CLaSp core logic

    # Statistics dict can be simplified or removed if not used by run_eval
    statistics = {
        "accept_length_list": [],  # Maybe track this
        # Add CLaSp specific args if needed by forward func or eval
        "skip_ratio": args.skip_ratio,
        "opt_interval": args.opt_interval,
        "draft_exit_threshold": args.draft_exit_threshold,
    }

    current_draft_skip_mask = None
    steps_since_last_optim = 0
    first_acc_rates = []

    # --- Run Evaluation ---
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=casspec_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        statistics=statistics,  # Pass simplified stats if needed
        logits_processor=logits_processor,
        args=args,  # Pass full args object to forward function
    )

    reorg_answer_file(answer_file)
    print("Evaluation finished.")
