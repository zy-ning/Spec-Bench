import argparse
import logging
import random
import time  # For timing checks

import numpy as np
import torch
from fastchat.utils import str_to_torch_dtype
from transformers import AutoTokenizer

# Assuming these are correctly adapted or available:
from evaluation.eval import reorg_answer_file, run_eval
from model.clasp.kv_cache import (
    # clone_past_key_values,
    initialize_past_key_values,
)
from model.clasp.modeling_llama import (
    LlamaForCausalLM,  # Ensure this class is adapted as per previous steps
)

# Import CLASP utils
from model.clasp.utils import (  # Ensure this points to your utils
    CLaSp_Skip_Layer_Strategy_SeqParallel,  # The core DP algorithm
    # cosine_similarity,  # Helper for DP if needed outside
    # normalize_tensor,  # Helper for DP if needed outside
    clasp_draft,
    clasp_verify,
    prepare_logits_processor,
    set_logger,
)


@torch.no_grad()
def clasp_forward(
    inputs,
    model,
    tokenizer,
    max_new_tokens,
    statistics=None,
    logits_processor=None,
    max_steps=512,
    args=None,
):
    """
    CLaSp forward pass without tree-based speculative decoding.
    Uses autoregressive drafting with layer skipping and parallel verification.
    """
    input_ids = inputs.input_ids.cuda()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    assert batch_size == 1, "Only support batch size 1 for now!!"
    # --- Initialization ---
    input_ids_list = input_ids.tolist()
    generated_token_count = 0
    total_steps = 0
    accept_length_list = []
    L = model.config.num_hidden_layers
    M = int(L * args.skip_ratio)  # Number of layers to skip (M for CLaSp DP)
    optim_interval_tokens = args.opt_interval
    K = args.draft_length_K
    DET = args.draft_exit_threshold
    # Initialize KV cache for the verify model
    # ! initialize_past_key_values now returns:
    # past_key_values (List[List[KVCache]]),
    # past_key_values_data (List[Tensor]),
    # current_length_data (Tensor)
    past_key_values, past_key_values_data, current_length_data = (
        initialize_past_key_values(model.model)
    )  # Pass base model
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    
    # --- Initial Prefill ---
    start_prefill = time.time()
    # The model's forward should internally use/update the KVCache objects
    # when use_cache=True. Pass the list of KVCache objects.
    prefill_outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,  # Pass the list of KVCache objects
        use_cache=True,
        output_hidden_states=True,
    )
    logging.info(f"Prefill time: {time.time() - start_prefill:.4f}s")
    # After prefill, past_key_values_list and current_length_data should be updated automatically
    # by the KVCache.cat method called inside the model's forward pass.
    input_len = input_ids.shape[1]  # Current length is updated in current_length_data
    # Get hidden states from the *last input token* for the *first* DP calc
    last_token_hidden_states = None
    if prefill_outputs.hidden_states:
        last_token_hidden_states = [
            s[:, -1, :].squeeze(0) for s in prefill_outputs.hidden_states
        ]
    current_draft_skip_mask = None
    tokens_accepted_since_last_optim = 0

    # --- Generation Loop ---
    timings = {
        "total_step": [],
        "dp_optim": [],
        "draft_clone_kv": [],
        "draft_loop": [],
        "avg_draft_time": [],
        "verify": [],
        "accept_update": [],
        "misc_overhead": [],
    }
    step_end_time = time.time()  # Initialize before loop

    while generated_token_count < max_new_tokens and total_steps < max_steps:
        total_steps += 1
        start_step = time.time()
        timings["misc_overhead"].append(start_step - step_end_time)
        # --- 1. Layer Optimization (Run periodically) ---

        dp_start_time = time.time()
        run_dp = False
        if last_token_hidden_states is not None and (
            current_draft_skip_mask is None
            or tokens_accepted_since_last_optim >= optim_interval_tokens
        ):
            start_dp = time.time()
            logging.info(f"Running CLaSp DP Optimization at step {total_steps}...")
            current_draft_skip_mask = CLaSp_Skip_Layer_Strategy_SeqParallel(
                L=L,
                M=M,
                hidden_states_H=last_token_hidden_states,
                model_layers=model.model.layers,
                device=device,
            )
            # logging.info("CLaSp DP Optimization: Layer to skip: ", current_draft_skip_mask)
            tokens_accepted_since_last_optim = 0  # Reset counter
            logging.info(
                f"CLaSp DP time: {time.time() - start_dp:.4f}s. Skipped {torch.sum(current_draft_skip_mask)} layers."
            )
        elif last_token_hidden_states is None:
            logging.warning(
                f"Skipping CLaSp DP at step {total_steps}: No hidden states available."
            )
            # Fallback: Use no skips or previous mask? Let's use previous or None.
            if current_draft_skip_mask is None:
                logging.warning(
                    "No skip mask calculated yet. Drafting with full model."
                )

        dp_end_time = time.time()
        if run_dp:
            timings["dp_optim"].append(dp_end_time - dp_start_time)

        # --- 2. Drafting (Autoregressive) ---
        start_draft = time.time()
        draft_tokens = []
        # # Use a *cloned* KV cache structure for drafting
        # # We need to clone the underlying data tensors and create new KVCache objects
        # draft_past_key_values_data_list = [d.clone() for d in past_key_values_data_list]
        # draft_current_length_data = (
        #     current_length_data.clone()
        # )  # This tracks lengths for the draft cache
        # # Create new KVCache objects pointing to the *cloned* data
        # draft_kv_cache_list = clone_past_key_values(
        #     model, draft_past_key_values_data_list, draft_current_length_data
        # )

        next_draft_input_ids = torch.tensor(
            [input_ids_list[0][-1]], device=device
        ).unsqueeze(0)

        clone_kv_end_time = time.time()
        timings["draft_clone_kv"].append(clone_kv_end_time - start_draft)

        for draft_step_idx in range(K):
            draft_i_start = time.time()
            # Pass the draft KVCache list to the model during drafting
            with model.self_draft(dynamic_skip_mask=current_draft_skip_mask):
                draft_outputs = model(  # Call the main model forward
                    input_ids=next_draft_input_ids,
                    # past_key_values=draft_kv_cache_list,  # Use draft cache list
                    past_key_values=past_key_values,
                    # use_cache=True,
                    output_hidden_states=False,
                )
            draft_i_end_time = time.time()
            logging.info(
                f"Drafting foward step {draft_step_idx + 1} time: {draft_i_end_time - draft_i_start:.4f}s"
            )
            draft_logits = draft_outputs.logits[:, -1, :]  # Logits for next token
            # draft_kv_cache_list is automatically updated by the forward pass
            # (DET check and sampling logic unchanged)
            if logits_processor:
                draft_logits_processed = logits_processor(
                    torch.tensor(input_ids_list, device=device), draft_logits
                )  # Pass full history?
            else:
                draft_logits_processed = draft_logits
            draft_probs = torch.softmax(draft_logits_processed, dim=-1)
            top1_prob = torch.max(draft_probs, dim=-1).values.item()
            if logits_processor:
                next_token = torch.multinomial(draft_probs, num_samples=1)
            else:
                next_token = torch.argmax(draft_logits_processed, dim=-1, keepdim=True)
            draft_tokens.append(next_token.item())
            next_draft_input_ids = next_token
            if top1_prob < DET and len(draft_tokens) > 0:
                logging.info(
                    f"Drafting stopped early at len {len(draft_tokens)} due to DET ({top1_prob:.3f} < {DET})"
                )
                break
            if next_token.item() == tokenizer.eos_token_id:
                break
            if len(draft_tokens) + generated_token_count >= max_new_tokens - 2:
                logging.info(
                    f"Drafting stopped due to max new tokens limit ({max_new_tokens})."
                )
                break
        
        # --- End Drafting Loop ---
        draft_loop_end_time = time.time()
        timings["draft_loop"].append(draft_loop_end_time - clone_kv_end_time)
        num_drafted = len(draft_tokens)
        timings["avg_draft_time"].append(
            (draft_loop_end_time - start_draft) / num_drafted if num_drafted > 0 else 0
        )
        logging.info(
            f"Drafting time: {draft_loop_end_time - start_draft:.4f}s. Drafted {num_drafted} tokens."
        )

        # --- 3. Verification (Parallel) ---
        start_verify = time.time()
        # verify_kv_cache_list = past_key_values_list  # Use main KVCache list
        current_seq_len = current_length_data[
            0
        ].item()  # Get current length BEFORE verify
        if num_drafted > 0:
            verify_input_list = [input_ids_list[0][-1]] + draft_tokens
            verify_input_ids = torch.tensor([verify_input_list], device=device)
        else:  # No tokens drafted, generate one with full model
            verify_input_ids = torch.tensor(
                [input_ids_list[0][-1]], device=device
            ).unsqueeze(0)
        # Run verification using the *full* model
        # Pass the main KVCache list
        verify_outputs = model(
            input_ids=verify_input_ids,
            past_key_values=past_key_values,  # Pass main cache list
            # use_cache=True,  # This should update the main cache IN PLACE
            output_hidden_states=True,
            output_attentions=False,
        )
        verify_logits = verify_outputs.logits
        verify_hidden_states = verify_outputs.hidden_states
        verify_end_time = time.time()
        timings["verify"].append(verify_end_time - start_verify)
        logging.info(f"Verification time: {verify_end_time - start_verify:.4f}s.")
        # NOTE: verify_kv_cache_list (past_key_values_list) is now updated by verify step

        # --- 4. Acceptance & Update ---
        start_accept = time.time()
        accepted_len = 0  # Number of *drafted* tokens accepted
        # rollback_len = num_drafted  # How many tokens in verify_kv_cache_list need potential rollback
        final_next_token = None
        last_accepted_token_hidden_states = None
        # Compare draft tokens with verify model predictions sequentially
        for k in range(num_drafted):
            pred_logits_k = verify_logits[:, k, :]
            if logits_processor:
                pred_logits_k = logits_processor(
                    torch.tensor(input_ids_list, device=device), pred_logits_k
                )
            verify_probs_k = torch.softmax(pred_logits_k, dim=-1)
            if logits_processor:
                verify_token_k = torch.multinomial(verify_probs_k, num_samples=1).item()
            else:
                verify_token_k = torch.argmax(pred_logits_k, dim=-1).item()
            if draft_tokens[k] == verify_token_k:
                # Accept draft token
                accepted_len += 1
                # Hidden states correspond to the token *producing* this prediction (index k)
                if verify_hidden_states:
                    try:
                        last_accepted_token_hidden_states = [
                            s[:, k, :].squeeze(0) for s in verify_hidden_states
                        ]
                    except IndexError:
                        last_accepted_token_hidden_states = None
                else:
                    last_accepted_token_hidden_states = None
            else:
                # Mismatch: Need to rollback KV cache and accept verify_token_k
                final_next_token = verify_token_k
                # rollback_len = k  # Rollback cache entries from index k onwards (relative to verify input)
                # Hidden states correspond to the token at the mismatch point (index k)
                if verify_hidden_states:
                    try:
                        last_accepted_token_hidden_states = [
                            s[:, k, :].squeeze(0) for s in verify_hidden_states
                        ]
                    except IndexError:
                        last_accepted_token_hidden_states = None
                else:
                    last_accepted_token_hidden_states = None
                break  # Stop acceptance loop
        # If all draft tokens accepted, sample the bonus token
        if accepted_len == num_drafted:
            # rollback_len = num_drafted  # No rollback needed, but length is num_drafted
            final_logits = verify_logits[
                :, num_drafted, :
            ]  # Logits after last draft token
            if logits_processor:
                final_logits = logits_processor(
                    torch.tensor(input_ids_list + [draft_tokens[-1]], device=device),
                    final_logits,
                )
            final_probs = torch.softmax(final_logits, dim=-1)
            if logits_processor:
                final_next_token = torch.multinomial(final_probs, num_samples=1).item()
            else:
                final_next_token = torch.argmax(final_logits, dim=-1).item()
            # Hidden states correspond to the bonus token prediction point (index num_drafted)
            if verify_hidden_states:
                try:
                    last_accepted_token_hidden_states = [
                        s[:, num_drafted, :].squeeze(0) for s in verify_hidden_states
                    ]
                except IndexError:
                    last_accepted_token_hidden_states = None
            else:
                last_accepted_token_hidden_states = None

        # --- Update Sequence and KV Cache Length ---
        # Add accepted draft tokens to sequence
        input_ids_list[0].extend(draft_tokens[:accepted_len])
        # Add the final token (either mismatch or bonus)
        input_ids_list[0].append(final_next_token)
        # Calculate the actual number of tokens added to the main KV cache by verify
        # This depends on the length of verify_input_ids
        verify_added_len = verify_input_ids.shape[1]
        # Calculate how many steps to rollback in the main KV cache
        rollback_steps = verify_added_len - (accepted_len + 1)
        # Update the current_length_data tensor to reflect the true length
        new_len = current_seq_len + accepted_len + 1
        current_length_data.fill_(new_len)  # Set length for ALL layers
        # Total accepted tokens in this verify step
        step_accept_len = accepted_len + 1
        accept_length_list.append(step_accept_len)
        generated_token_count += step_accept_len
        tokens_accepted_since_last_optim += step_accept_len
        accept_update_end_time = time.time()
        timings["accept_update"].append(accept_update_end_time - start_accept)
        logging.info(
            f"Acceptance time: {time.time() - start_accept:.4f}s. Accepted {step_accept_len} tokens. Rolled back {rollback_steps} KV entries."
        )
        logging.info(
            f"Step {total_steps} time: {time.time() - start_step:.4f}s | Accepted: {step_accept_len} | Total Gen: {generated_token_count}"
        )
        
        step_end_time = time.time()
        timings["total_step"].append(step_end_time - start_step)
        # --- Check for EOS ---
        if tokenizer.eos_token_id == final_next_token:
            logging.info("EOS token generated. Stopping.")
            break

    # --- Final Output ---
    # (Output logic unchanged)
    output_ids = torch.tensor(
        [input_ids_list[0][input_len:]], device=device
    )  # Generated part
    avg_accept_len = np.mean(accept_length_list) if accept_length_list else 0
    logging.info(
        f"Finished. Total steps: {total_steps}, Total generated: {generated_token_count}, Avg accept/step: {avg_accept_len:.2f}"
    )
    
    # --- Print Timings ---
    logging.info("--- Performance Timings (Average per Step) ---")
    for key, values in timings.items():
        if values:
             avg_time = np.mean(values)
             logging.info(f"{key}: {avg_time:.4f}s")
        else:
             logging.info(f"{key}: N/A (not run or no steps)")
    logging.info(f"Average Acceptance Length: {avg_accept_len:.2f}")
    logging.info(f"Finished. Total steps: {total_steps}, Total generated: {generated_token_count}")

    return output_ids, generated_token_count, total_steps, accept_length_list


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
    # --- Keep essential args from SWIFT ---
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

    # --- Add/Modify args for CLaSp ---
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
        + "-clasp-"
        + str(args.dtype)
        + "-temp-"
        + str(args.temperature)
        + "-topp-"
        + str(args.top_p)
        + "-seed-"
        + str(args.seed)
        + "-maxntok-"
        + str(args.max_new_tokens)
        + f"-I{args.opt_interval}"
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

    # --- Run Evaluation ---
    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=clasp_forward,  # Use the CLaSp forward function
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,  # CLaSp paper doesn't mention multiple choices
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        statistics=statistics,  # Pass simplified stats if needed
        logits_processor=logits_processor,
        args=args,  # Pass full args object to forward function
    )

    reorg_answer_file(answer_file)
    print("Evaluation finished.")
