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
from model.casspec.kv_cache import initialize_past_key_values
from model.casspec.modeling_llama import LlamaForCausalLM
from model.casspec.utils import (
    evaluate_posterior,
    generate_candidates,
    generate_swift_buffers,
    get_cache_configuration,
    get_choices_list,
    initialize_swift,
    prepare_logits_processor,
    reset_swift_mode,
    set_logger,
    swift_draft,
    swift_optimization,
    tree_decoding,
    update_inference_inputs,
)

first_acc_rates = []

import copy

from model.casspec.utils import find_candidate_pred_tokens  # Add this import


def add_pld_paths_to_swift_buffers(
    swift_buffers,
    input_ids,
    sample_token,
    swift_logits,
    pld_max_ngram_size=3,
    pld_num_pred_tokens=7,
    max_pld_depths=3,
):
    """
    Add PLD candidate paths to existing SWIFT buffers.

    Args:
        swift_buffers: Original SWIFT buffers
        input_ids: Current input sequence
        sample_token: First greedy token from SWIFT
        swift_logits: SWIFT draft logits (tokens, probs, ops)
        pld_max_ngram_size: Max n-gram size for PLD matching
        pld_num_pred_tokens: Number of tokens PLD should predict
        max_pld_depths: Maximum tree depth to add PLD paths (0=root, 1=after 1st token, etc.)

    Returns:
        enhanced_candidates: Extended candidate tensor with PLD tokens
        enhanced_buffers: Updated SWIFT buffers with PLD paths added
    """
    device = swift_buffers["tree_indices"].device
    retrieve_indices = swift_buffers["retrieve_indices"]
    tree_indices = swift_buffers["tree_indices"]

    # Prepare to collect PLD tokens and paths
    pld_token_sequences = []
    pld_retrieve_paths = []

    # Get SWIFT tokens for building extended contexts
    swift_tokens = swift_logits[0]  # Shape: [num_draft_tokens, TOPK]

    # --- Generate PLD candidates at different depths ---
    for depth in range(min(max_pld_depths, len(swift_tokens) + 1)):
        if depth == 0:
            # PLD from current position (root)
            context = input_ids
            prefix_tokens = []
        else:
            # PLD after following greedy path to this depth
            # Greedy path = first token from sample_token + first tokens from swift_tokens
            if depth == 1:
                greedy_tokens = sample_token[0]  # First token
            else:
                # First token + subsequent greedy selections
                greedy_tokens = torch.cat(
                    [
                        sample_token[0],
                        swift_tokens[: depth - 1, 0],  # First choice at each level
                    ]
                )

            context = torch.cat([input_ids, greedy_tokens.unsqueeze(0)], dim=1)
            prefix_tokens = greedy_tokens.tolist()

        # Find PLD candidates from this context
        pld_tokens = find_candidate_pred_tokens(
            context,
            max_ngram_size=pld_max_ngram_size,
            num_pred_tokens=pld_num_pred_tokens,
        )

        if len(pld_tokens) > 0:
            pld_token_sequences.append(
                {"depth": depth, "tokens": pld_tokens, "prefix": prefix_tokens}
            )

    if not pld_token_sequences:
        # No PLD candidates found, return original
        return None, swift_buffers

    # --- Build enhanced candidate tensor ---
    # Original: [sample_token, swift_draft_tokens...]
    # Enhanced: [sample_token, swift_draft_tokens..., pld_tokens_depth0, pld_tokens_depth1, ...]

    base_candidates = torch.cat([sample_token[0], swift_tokens.view(-1)], dim=0)

    pld_all_tokens = []
    for pld_seq in pld_token_sequences:
        pld_all_tokens.append(pld_seq["tokens"])

    if pld_all_tokens:
        enhanced_candidates = torch.cat(
            [base_candidates, torch.cat(pld_all_tokens, dim=0)], dim=0
        )
    else:
        enhanced_candidates = base_candidates

    # --- Build PLD retrieve paths ---
    current_tree_size = len(tree_indices)
    pld_token_offset = len(
        base_candidates
    )  # Where PLD tokens start in enhanced_candidates
    max_path_len = retrieve_indices.shape[1]

    for pld_seq in pld_token_sequences:
        depth = pld_seq["depth"]
        tokens = pld_seq["tokens"]

        # Build path: [0, greedy_to_depth..., pld_token_0, pld_token_1, ...]
        if depth == 0:
            # Path from root: [0, pld_token_idx, pld_token_idx+1, ...]
            base_path = [0]
        else:
            # Path following greedy: [0, 1, swift_greedy_indices..., pld_start]
            # We need to find indices in tree_indices for the greedy path
            greedy_path = [0]  # Root
            for d in range(depth):
                # The greedy choice at depth d is at position (sum of previous depths + 0)
                # This is simplified - you may need to track actual tree positions
                greedy_path.append(d + 1)  # Approximate greedy positions
            base_path = greedy_path

        # Add PLD token positions
        for i, _ in enumerate(tokens):
            path = base_path + [pld_token_offset + i]
            # Pad to max length
            path = path + [-1] * (max_path_len - len(path))
            pld_retrieve_paths.append(path[:max_path_len])

        pld_token_offset += len(tokens)

    # --- Update retrieve_indices ---
    if pld_retrieve_paths:
        pld_paths_tensor = torch.tensor(
            pld_retrieve_paths, dtype=torch.long, device=device
        )
        enhanced_retrieve_indices = torch.cat(
            [retrieve_indices, pld_paths_tensor], dim=0
        )

        enhanced_buffers = copy.deepcopy(swift_buffers)
        enhanced_buffers["retrieve_indices"] = enhanced_retrieve_indices

        # Extend p_indices and b_indices for new paths
        for _ in range(len(pld_retrieve_paths)):
            enhanced_buffers["p_indices"].append([-1] + [0] * (max_path_len - 1))
            enhanced_buffers["b_indices"].append([[] for _ in range(max_path_len)])

        logging.info(f"Added {len(pld_retrieve_paths)} PLD paths to SWIFT tree")
        return enhanced_candidates, enhanced_buffers

    return None, swift_buffers


def generate_candidates_with_pld(
    swift_logits,
    tree_indices,
    retrieve_indices,
    sample_token,
    logits_processor,
    enhanced_candidates=None,
):
    """
    Modified version of generate_candidates that uses enhanced candidates if available.
    """
    sample_token = sample_token.to(tree_indices.device)

    if enhanced_candidates is not None:
        # Use PLD-enhanced candidates
        candidates = enhanced_candidates
    else:
        # Original SWIFT candidates
        candidates_logit = sample_token[0]
        candidates_swift_logits = swift_logits[0]
        candidates = torch.cat(
            [candidates_logit, candidates_swift_logits.view(-1)], dim=-1
        )

    # Map to tree structure
    tree_candidates = candidates[tree_indices]
    tree_candidates_ext = torch.cat(
        [
            tree_candidates,
            torch.zeros((1), dtype=torch.long, device=tree_candidates.device),
        ],
        dim=0,
    )

    # Retrieve cartesian candidates
    cart_candidates = tree_candidates_ext[retrieve_indices]

    # Handle probabilities
    if logits_processor is not None:
        candidates_tree_prob = swift_logits[1]
        candidates_prob = torch.cat(
            [
                torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32),
                candidates_tree_prob.view(-1),
            ],
            dim=-1,
        )

        # Extend with dummy probs for PLD tokens
        if enhanced_candidates is not None:
            pld_token_count = len(enhanced_candidates) - len(candidates_prob)
            if pld_token_count > 0:
                pld_probs = (
                    torch.ones(
                        pld_token_count,
                        device=candidates_prob.device,
                        dtype=torch.float32,
                    )
                    * 0.5
                )  # Assign moderate confidence to PLD tokens
                candidates_prob = torch.cat([candidates_prob, pld_probs], dim=0)

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [
                tree_candidates_prob,
                torch.ones(
                    (1), dtype=torch.float32, device=tree_candidates_prob.device
                ),
            ],
            dim=0,
        )
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None

    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def swift_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    statistics=None,
    optimizer=None,
    utility=None,
    logits_processor=None,
    max_steps=512,
    use_pld=True,
    pld_max_ngram_size=3,
    pld_num_pred_tokens=7,
    max_pld_depths=2,
):
    """
    Enhanced SWIFT forward with PLD integration.

    New args:
        use_pld: Whether to enable PLD candidate augmentation
        pld_max_ngram_size: Max n-gram size for PLD
        pld_num_pred_tokens: Number of tokens PLD predicts
        max_pld_depths: Maximum tree depths to add PLD (0=root, 1=after 1st token, etc.)
    """
    input_ids = inputs.input_ids.cuda()
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
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
    swift_logits, sample_token, top1_prob = initialize_swift(
        input_ids,
        model,
        max_new_tokens,
        past_key_values,
        past_key_values_data,
        current_length_data,
        logits_processor=logits_processor,
    )
    logging.info(f"Prefill time: {time.time() - start_prefill:.4f}s")

    # Clone for optimization
    input_past_key_values_data = [kv.clone() for kv in past_key_values_data]
    input_current_length_data = current_length_data.clone()
    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    pld_contribution = 0  # Track how many tokens accepted from PLD
    timings = {
        "total_step": [],
        "draft_loop": [],
        "avg_draft_time": [],
        "verify": [],
        "accept_update": [],
        "misc_overhead": [],
        "pld_augmentation": [],  # New timing
    }
    step_end_time = time.time()
    for idx in range(max_steps):
        start_step = time.time()
        timings["misc_overhead"].append(start_step - step_end_time)
        draft_token_num += len(top1_prob)

        # Initialize SWIFT buffer
        swift_choices = eval(
            f"{get_choices_list(top1_prob, logits_processor=logits_processor)}"
        )
        swift_buffers = generate_swift_buffers(
            swift_choices, device=model.model.layers[-1].self_attn.q_proj.weight.device
        )
        model.swift_buffers = swift_buffers
        model.swift_choices = swift_choices
        model.model.swift_mask = swift_buffers["swift_attn_mask"]
        # --- PLD AUGMENTATION ---
        pld_start = time.time()
        enhanced_candidates = None
        active_buffers = swift_buffers

        if use_pld and input_ids.shape[1] >= pld_max_ngram_size:
            enhanced_candidates, enhanced_buffers = add_pld_paths_to_swift_buffers(
                swift_buffers,
                input_ids,
                sample_token,
                swift_logits,
                pld_max_ngram_size=pld_max_ngram_size,
                pld_num_pred_tokens=pld_num_pred_tokens,
                max_pld_depths=max_pld_depths,
            )

            if enhanced_candidates is not None:
                active_buffers = enhanced_buffers
                logging.info(
                    f"Step {idx}: Enhanced with PLD, "
                    f"paths: {active_buffers['retrieve_indices'].shape[0]} "
                    f"(+{active_buffers['retrieve_indices'].shape[0] - swift_buffers['retrieve_indices'].shape[0]})"
                )

        timings["pld_augmentation"].append(time.time() - pld_start)

        # Generate candidates (now possibly PLD-enhanced)
        candidates, cart_candidates_prob, tree_candidates = (
            generate_candidates_with_pld(
                swift_logits,
                active_buffers["tree_indices"],
                active_buffers["retrieve_indices"],
                sample_token,
                logits_processor,
                enhanced_candidates=enhanced_candidates,
            )
        )
        # Tree decoding (verification)
        logits, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            swift_buffers["swift_position_ids"],  # Use original position IDs
            input_ids,
            active_buffers[
                "retrieve_indices"
            ],  # Use possibly enhanced retrieve indices
        )
        verify_end_time = time.time()
        timings["verify"].append(verify_end_time - pld_start)  # Includes PLD time
        # Evaluate posterior
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits,
            candidates,
            logits_processor,
            cart_candidates_prob,
            swift_logits[2],
            active_buffers["p_indices"],
            tree_candidates,
            active_buffers["b_indices"],
        )
        # Track if PLD contributed
        if (
            enhanced_candidates is not None
            and best_candidate >= swift_buffers["retrieve_indices"].shape[0]
        ):
            pld_contribution += accept_length + 1
            logging.info(f"✓ PLD path accepted! Length: {accept_length + 1}")
        if accept_length == 0:
            first_acc_rates.append(0)
        else:
            first_acc_rates.append(1)
        # Update inputs
        input_ids, new_token_num, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            active_buffers["retrieve_indices"],  # Use active buffers
            logits_processor,
            new_token_num,
            past_key_values_data,
            current_length_data,
            sample_p,
        )
        accept_update_end_time = time.time()
        timings["accept_update"].append(accept_update_end_time - verify_end_time)
        # Layer set optimization
        if (
            new_token_num > (statistics["context_window"] + 1)
            and statistics["optimization"]
            and idx % statistics["opt_interval"] == 0
        ):
            logging.info("Swift optimization" + "-" * 10)
            swift_optimization(
                model,
                input_ids[:, input_len:],
                input_past_key_values_data,
                input_current_length_data,
                new_token_num,
                statistics,
                optimizer=optimizer,
                utility=utility,
            )
        # SWIFT drafting for next iteration
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
            draft_time / len(top1_prob) if len(top1_prob) > 0 else 0
        )
        # Update acceptance stats
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1

        # Stopping conditions
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break
        step_end_time = time.time()
        timings["total_step"].append(step_end_time - start_step)
    total_acc_rate = total_acc_num / draft_token_num if draft_token_num > 0 else 0
    logging.info(f"Total acceptance rate: {total_acc_rate:.4f}")
    logging.info(
        f"PLD contribution: {pld_contribution}/{new_token_num} tokens ({pld_contribution / new_token_num * 100:.2f}%)"
    )
    logging.info(f"Total steps: {idx}")

    # Print timings
    logging.info("--- Performance Timings (Average per Step) ---")
    for key, values in timings.items():
        if values:
            avg_time = np.mean(values)
            logging.info(f"{key}: {avg_time:.4f}s")
        else:
            logging.info(f"{key}: N/A")

    logging.info(f"Average Acceptance Length: {np.mean(accept_length_list) - 1:.4f}")
    logging.info(f"Finished. Total steps: {idx}, Total generated: {new_token_num}")
    return input_ids, new_token_num, idx + 1, accept_length_list


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
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--use-pld",
        action="store_true",
        default=False,
        help="Enable PLD candidate augmentation"
    )
    parser.add_argument(
        "--pld-max-ngram",
        type=int,
        default=3,
        help="Max n-gram size for PLD"
    )
    parser.add_argument(
        "--pld-num-tokens",
        type=int,
        default=7,
        help="Number of tokens PLD predicts"
    )
    parser.add_argument(
        "--pld-max-depths",
        type=int,
        default=2,
        help="Maximum tree depths to add PLD paths"
    )

    args = parser.parse_args()

    args.model_name = (
        args.model_id
        + "-casspec-"
        + str(args.dtype)
        + "-temp-"
        + str(args.temperature)
        + "-top-p-"
        + str(args.top_p)
        + "-seed-"
        + str(args.seed)
        + "-max_new_tokens-"
        + str(args.max_new_tokens)
        + "-opt_interval-"
        + str(args.opt_interval)
        + "-bayes_interval-"
        + str(args.bayes_interval)
        + "-max_opt-"
        + str(args.max_opt_iter)
        + "-max_tolerance-"
        + str(args.max_tolerance_iter)
        + "-max_score-"
        + str(args.max_score)
        + "-context_window-"
        + str(args.context_window)
        + "-skip_ratio-"
        + str(args.skip_ratio)
        # + "-pld-" if args.use_pld else ""
        # + f"maxngram{args.pld_max_ngram}-numtok{args.pld_num_tokens}-maxdepth{args.pld_max_depths}" if args.use_pld else ""
    )
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
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(
            temperature=args.temperature, top_p=args.top_p
        )
    else:
        logits_processor = None

    if args.cache_hit:
        # Load the cached layer set configuration
        args.optimization, args.bayes = False, False
        _attn_skip_layer_id_set, _mlp_skip_layer_id_set = get_cache_configuration(
            model_name=args.model_id, bench_name=args.bench_name
        )
    else:
        # Unified layer set initialization
        _attn_skip_layer_id_set = np.arange(
            1, model.config.num_hidden_layers - 1, 2
        )  # keep the first and last layer
        _mlp_skip_layer_id_set = np.arange(1, model.config.num_hidden_layers - 1, 2)

    model.set_skip_layers(_attn_skip_layer_id_set, _mlp_skip_layer_id_set)

    seed_everything(args.seed)
    # Bayes Optimization Settings
    pbounds = {
        f"x{i}": (0, 1) for i in range((model.config.num_hidden_layers - 2) * 2)
    }  # keep the first and last layer
    optimizer = BayesianOptimization(
        f=None, pbounds=pbounds, random_state=1, verbose=1, allow_duplicate_points=True
    )
    optimizer.set_gp_params(alpha=1e-2)
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    statistics = {
        "origin_score": 0,
        "opt_iter": 0,
        "tolerance_iter": 0,
        "skip_ratio": args.skip_ratio,
        "acceptance_rate_list": [],
        "opt_interval": args.opt_interval,
        "bayes_interval": args.bayes_interval,
        "max_opt_iter": args.max_opt_iter,
        "max_tolerance_iter": args.max_tolerance_iter,
        "max_score": args.max_score,
        "context_window": args.context_window,
        "optimization": args.optimization,
        "bayes": args.bayes,
    }

    def swift_forward_wrapper(inputs, model, tokenizer, max_new_tokens, **kwargs):
        return swift_forward(
            inputs, model, tokenizer, max_new_tokens,
            use_pld=args.use_pld,
            pld_max_ngram_size=args.pld_max_ngram,
            pld_num_pred_tokens=args.pld_num_tokens,
            max_pld_depths=args.pld_max_depths,
            **kwargs
        )


    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=swift_forward_wrapper,
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
