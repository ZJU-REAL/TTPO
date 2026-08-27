import os
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    GRPOTrainer,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field

from ttpo_voting import extract_boxed_answer, majority_vote, answers_equivalent

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class TTRLScriptArguments(ScriptArguments):
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name) to log runs under."},
    )
    wandb_project: str = field(
        default="ttrl-training",
        metadata={"help": "WandB project name to log runs under."},
    )
    min_consensus_count: int = field(
        default=2,
        metadata={
            "help": "Minimum majority-cluster size for a reward signal to be assigned. "
            "Below this threshold, all rewards for that prompt group are 0.0."
        },
    )
    val_files: str = field(
        default=None,
        metadata={"help": "Validation dataset path."}
    )
    val_n: int = field(
        default=12,
        metadata={"help": "Number of completions to sample per problem during evaluation."},
    )
    test_freq: int = field(
        default=20,
        metadata={"help": "Run evaluation every N training steps."},
    )


def make_reward_ttrl(num_generations, min_consensus_count=2):
    """Factory that returns a TTRL reward function using majority voting.

    In TRL 0.26 colocate mode, the reward function is called per-rank with only
    local completions. We all-gather completions across ranks, group them by prompt,
    perform majority voting on the full group, then return only the local rewards.
    """
    import torch
    import torch.distributed as dist
    from collections import OrderedDict

    def reward_ttrl(completions, **kwargs):
        gt_answers = kwargs.get("Answer")
        prompts = kwargs.get("prompts")
        n = num_generations
        local_size = len(completions)

        # All-gather completions and prompts across ranks
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            all_completions_gathered = [None] * world_size
            dist.all_gather_object(all_completions_gathered, completions)
            all_completions = [c for rank_comps in all_completions_gathered for c in rank_comps]

            local_prompts = prompts if prompts is not None else [None] * local_size
            all_prompts_gathered = [None] * world_size
            dist.all_gather_object(all_prompts_gathered, local_prompts)
            all_prompts = [p for rank_ps in all_prompts_gathered for p in rank_ps]

            if gt_answers is not None:
                all_gt_gathered = [None] * world_size
                dist.all_gather_object(all_gt_gathered, gt_answers)
                all_gt = [g for rank_gs in all_gt_gathered for g in rank_gs]
            else:
                all_gt = None
        else:
            world_size = 1
            rank = 0
            all_completions = completions
            all_prompts = prompts if prompts is not None else [None] * local_size
            all_gt = gt_answers

        # Group completions by prompt (hash for efficiency)
        prompt_groups = OrderedDict()
        for i, p in enumerate(all_prompts):
            key = hash(p) if isinstance(p, str) else hash(str(p))
            if key not in prompt_groups:
                prompt_groups[key] = []
            prompt_groups[key].append(i)

        # Compute rewards for all completions via majority voting
        all_rewards = [0.0] * len(all_completions)
        vote_correct = 0
        vote_total = 0
        total_extractable = 0
        consensus_sum = 0
        majority_ratio_sum = 0.0
        prompts_with_consensus = 0
        pass_at_k_hits = 0
        pass_at_k_total = 0
        num_prompts = len(prompt_groups)

        for key, indices in prompt_groups.items():
            chunk = [all_completions[i] for i in indices]
            group_n = len(chunk)
            answers = [extract_boxed_answer(c) for c in chunk]
            pseudo_label, correct_mask, consensus_count = majority_vote(answers)

            extractable = sum(1 for a in answers if a is not None and a != "")
            total_extractable += extractable
            consensus_sum += consensus_count
            majority_ratio_sum += consensus_count / group_n

            if consensus_count >= min_consensus_count and pseudo_label:
                group_rewards = [1.0 if m else 0.0 for m in correct_mask]
                prompts_with_consensus += 1
            else:
                group_rewards = [0.0] * group_n

            for idx, r in zip(indices, group_rewards):
                all_rewards[idx] = r

            if all_gt is not None:
                gt = all_gt[indices[0]]
                if gt and pseudo_label:
                    vote_total += 1
                    if answers_equivalent(pseudo_label, gt):
                        vote_correct += 1
                if gt:
                    pass_at_k_total += 1
                    if any(a and answers_equivalent(a, gt) for a in answers):
                        pass_at_k_hits += 1

        # Return only the local rank's slice of rewards
        local_start = rank * local_size
        local_rewards = all_rewards[local_start:local_start + local_size]

        # Log global metrics (no all-reduce needed since we already gathered everything)
        total_completions = len(all_completions)
        reward_sum = float(sum(all_rewards))
        positive_count = sum(1 for r in all_rewards if r > 0)

        metrics = {
            "ttrl/consensus_mean": consensus_sum / num_prompts if num_prompts > 0 else 0,
            "ttrl/majority_ratio": majority_ratio_sum / num_prompts if num_prompts > 0 else 0,
            "ttrl/prompts_with_consensus": prompts_with_consensus,
            "ttrl/prompts_without_consensus": num_prompts - prompts_with_consensus,
            "ttrl/extractable_fraction": total_extractable / total_completions if total_completions > 0 else 0,
            "ttrl/reward_mean": reward_sum / total_completions if total_completions > 0 else 0,
            "ttrl/positive_fraction": positive_count / total_completions if total_completions > 0 else 0,
        }
        if vote_total > 0:
            metrics["ttrl/vote_accuracy"] = vote_correct / vote_total
        if pass_at_k_total > 0:
            metrics[f"ttrl/pass_at_{n}"] = pass_at_k_hits / pass_at_k_total

        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            if wandb.run is not None:
                wandb.log(metrics)
            print(
                f"[TTRL-Global] P={num_prompts}, n={n}, "
                f"extractable={metrics['ttrl/extractable_fraction']:.2%}, "
                f"consensus={prompts_with_consensus}/{num_prompts}, "
                f"consensus_mean={metrics['ttrl/consensus_mean']:.1f}, "
                f"majority_ratio={metrics['ttrl/majority_ratio']:.2%}, "
                f"reward_mean={metrics['ttrl/reward_mean']:.3f}, "
                f"positive={metrics['ttrl/positive_fraction']:.2%}"
                + (f", vote_acc={vote_correct}/{vote_total} ({metrics['ttrl/vote_accuracy']:.2%})" if vote_total > 0 else "")
                + (f", pass@{n}={metrics[f'ttrl/pass_at_{n}']:.2%}" if pass_at_k_total > 0 else "")
            )

        return local_rewards

    return reward_ttrl


def make_format_prompt(tokenizer, enable_thinking=True):
    def format_prompt(example):
        messages = [
            {
                "role": "user",
                "content": f"Problem: {example.get('Question', example.get('problem'))}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        answer = example.get("Answer", example.get("answer"))
        return {"prompt": prompt, "Answer": None if answer is None else str(answer)}

    return format_prompt


def run_ttrl_val(trainer, eval_dataset, val_n, global_step):
    import torch
    import torch.distributed as dist
    from vllm import SamplingParams
    from tqdm import tqdm

    if not (trainer.args.use_vllm and getattr(trainer, "vllm_mode", None) == "colocate"):
        return {}

    is_main = trainer.accelerator.is_main_process
    rank = trainer.accelerator.process_index
    world_size = trainer.accelerator.num_processes

    # All ranks: wake vLLM and sync latest trained weights
    if trainer.args.vllm_enable_sleep_mode:
        torch.cuda.empty_cache()
        trainer.llm.wake_up(tags=["weights"])
        trainer.llm.wake_up(tags=["kv_cache"])

    trainer._move_model_to_vllm()

    # Shard eval dataset across ranks (interleaved)
    all_prompts = eval_dataset["prompt"]
    all_gt_answers = eval_dataset["Answer"]
    local_indices = list(range(rank, len(all_prompts), world_size))
    local_prompts = [all_prompts[i] for i in local_indices]
    local_gt_answers = [all_gt_answers[i] for i in local_indices]

    sampling_params = SamplingParams(
        n=val_n,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_tokens=3072,
    )

    # Generate in batches (same batch size as training)
    batch_size = trainer.args.per_device_train_batch_size
    all_outputs = []
    num_batches = (len(local_prompts) + batch_size - 1) // batch_size

    iterator = range(0, len(local_prompts), batch_size)
    if is_main:
        iterator = tqdm(
            iterator,
            desc=f"[Eval step {global_step}] {len(all_prompts)} problems / {world_size} GPUs",
            total=num_batches,
            unit="batch",
        )

    for start in iterator:
        batch_prompts = local_prompts[start : start + batch_size]
        batch_outputs = trainer.llm.generate(
            batch_prompts, sampling_params=sampling_params, use_tqdm=False
        )
        all_outputs.extend(batch_outputs)

    # All ranks: sleep vLLM
    if trainer.args.vllm_enable_sleep_mode:
        trainer.llm.sleep(level=2)

    # Compute local metrics
    pass_count = 0
    avg_sum = 0.0
    n_valid = 0
    for i, output in enumerate(all_outputs):
        gt = local_gt_answers[i]
        if not gt:
            continue
        n_valid += 1
        extracted = [extract_boxed_answer(o.text) for o in output.outputs]
        correct = sum(1 for a in extracted if a and answers_equivalent(a, gt))
        avg_sum += correct / val_n
        if correct > 0:
            pass_count += 1

    # All-reduce metrics across ranks
    stats = torch.tensor(
        [float(pass_count), float(n_valid), avg_sum],
        dtype=torch.float64, device="cuda",
    )
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    g_pass_count = int(stats[0].item())
    g_n_valid = int(stats[1].item())
    g_avg_sum = stats[2].item()

    metrics = {}
    if g_n_valid > 0:
        metrics = {
            f"eval/avg@{val_n}": g_avg_sum / g_n_valid,
            f"eval/pass@{val_n}": g_pass_count / g_n_valid,
        }
        if is_main:
            if wandb.run is not None:
                wandb.log({**metrics, "eval/step": global_step})
            print(
                f"[TTRL-Eval] step={global_step}, "
                f"avg@{val_n}={metrics[f'eval/avg@{val_n}']:.4f}, "
                f"pass@{val_n}={metrics[f'eval/pass@{val_n}']:.4f} "
                f"({g_pass_count}/{g_n_valid})"
            )

    return metrics


if __name__ == "__main__":
    parser = TrlParser((TTRLScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_name = (
            f"TTRL_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )

    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION (TTRL)")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Prompt Length: {training_args.max_prompt_length}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"Min Consensus Count: {script_args.min_consensus_count}")
    print(f"Reward: majority-vote pseudo-labels (no ground truth)")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_prompt_length": training_args.max_prompt_length,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "min_consensus_count": script_args.min_consensus_count,
                "method": "TTRL (majority-vote pseudo-labels)",
            },
        )

    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_files = load_dataset(script_args.dataset_name)
    eval_files = load_dataset(script_args.val_files)
    train_dataset = train_files["train"]
    eval_dataset = eval_files["train"]

    train_dataset = train_dataset.map(make_format_prompt(tokenizer), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(
        make_format_prompt(tokenizer, enable_thinking=False), remove_columns=eval_dataset.column_names,
    )

    reward_fn = make_reward_ttrl(
        num_generations=training_args.num_generations,
        min_consensus_count=script_args.min_consensus_count,
    )

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Custom eval callback
    if script_args.val_files and script_args.test_freq > 0:
        from transformers import TrainerCallback

        val_n = script_args.val_n
        test_freq = script_args.test_freq

        class TTRLValCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step > 0 and state.global_step % test_freq == 0:
                    run_ttrl_val(trainer, eval_dataset, val_n, state.global_step)

        trainer.add_callback(TTRLValCallback())

    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if checkpoints:
            resume_from_checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
