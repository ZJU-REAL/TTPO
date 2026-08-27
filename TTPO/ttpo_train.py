import os
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from ttpo_trainer import TTPOTrainer
from dataclasses import dataclass, field

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class TTPOScriptArguments(ScriptArguments):
    run_config: str = field(
        default=None,
        metadata={"help": "Run name for this experiment."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty for vLLM generation."},
    )
    num_rollouts: int = field(
        default=8,
        metadata={"help": "K_gen: how many rollouts vLLM samples per prompt. Should be >= num_train_rollouts; oversampling buffers against empty-answer rollouts."},
    )
    num_train_rollouts: int = field(
        default=8,
        metadata={"help": "K_train: how many rollouts per prompt actually enter the loss. Must be <= num_rollouts. Selection prefers extractable answers; backfills with shortest empty-answer rollouts when needed."},
    )
    min_consensus_count: int = field(
        default=2,
        metadata={"help": "Minimum majority-cluster size for full contrastive update. Below this threshold, only the shortest valid-answer completion is treated as positive."},
    )
    use_gt: bool = field(
        default=False,
        metadata={
            "help": "Use the ground-truth answer as the pseudo-label instead of majority voting. "
                    "Bypasses --min_consensus_count (GT is always trusted), so positives are exactly "
                    "the rollouts equivalent to GT. Errors out if a sample has no ground truth."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={"help": "Clip per-token JSD to this max value. Set to 0 for no clipping."},
    )
    jsd_clip_mode: str = field(
        default="element",
        metadata={"help": "Where to apply jsd_token_clip: 'element' = before sum over vocab (per vocab entry), 'token' = after sum over vocab (per token position)."},
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Run the answer-conditioned teacher forward under an EMA snapshot of the "
            "student. Provides a stable anchor that lags the live student."
        },
    )
    ema_decay: float = field(
        default=0.995,
        metadata={"help": "EMA decay for the teacher snapshot: ema = decay*ema + (1-decay)*student."},
    )
    fixed_teacher: bool = field(
        default=True,
        metadata={
            "help": "Run the answer-conditioned teacher forward with LoRA adapters disabled "
            "(base model = frozen initial policy). Requires --use_peft. Mutually exclusive "
            "with --use_ema_teacher and --periodic_sync_teacher."
        },
    )
    periodic_sync_teacher: bool = field(
        default=False,
        metadata={
            "help": "Hard-copy student weights into the teacher snapshot every "
            "--teacher_sync_interval steps. Between syncs the teacher is frozen. "
            "Mutually exclusive with --use_ema_teacher and --fixed_teacher."
        },
    )
    teacher_sync_interval: int = field(
        default=10,
        metadata={"help": "How often (in optimizer steps) to sync the teacher snapshot from the student. Only used when --periodic_sync_teacher is set."},
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (matches the main OPSD setup: student rolls out without <think>)."
        },
    )
    teacher_thinking: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the answer-conditioned teacher "
            "forward. Default True. Set to False for the matched non-thinking ablation (both nonthink)."
        },
    )
    kl_mode: float = field(
        default=-3,
        metadata={
            "help": "KL direction control. In [0,1]: generalized JSD (0=fkl, 1=rkl, between=JSD mixture). "
                    "-1: pos=fkl neg=rkl. -2: pos=rkl neg=fkl. "
                    "-3: pos=fkl, neg=grpo policy gradient. "
                    "-4: pos=grpo policy gradient, neg=fkl."
        },
    )
    max_tokens: int = field(
        default=1024,
        metadata={
            "help": "Maximum number of completion tokens that participate in the gradient update. "
                    "When set, completions are truncated/padded to this length for training, while "
                    "max_completion_length still controls vLLM sampling length. Defaults to max_completion_length."
        },
    )
    positive_fraction: float = field(
        default=0.5,
        metadata={
            "help": "Positive/negative ratio among selected K_train rollouts. "
                    "-1 (default): dynamic sqrt-tempered ratio based on empirical pos/neg counts. "
                    "-2: naive first-K_train selection (no pos/neg split, no quality filtering). "
                    "0~1: exact positive fraction (e.g. 0.5 = 50%% positive + 50%% negative, backfill if short)."
        },
    )
    pos_select: int = field(
        default=0,
        metadata={
            "help": "Selection strategy for positive samples within their quota. "
                    "0: prefer shortest extractable trajectories. "
                    "1: prefer highest mean per-token KL(teacher || student). "
                    "2: prefer highest mean per-token student log-probability. "
                    "3: prefer longest extractable trajectories."
        },
    )
    neg_select: int = field(
        default=0,
        metadata={
            "help": "Selection strategy for negative samples within their quota. "
                    "0: prefer shortest extractable trajectories. "
                    "1: prefer highest mean per-token KL(teacher || student). "
                    "2: prefer highest mean per-token student log-probability. "
                    "3: prefer longest extractable trajectories."
        },
    )
    token_weighting_retention: float = field(
        default=0.0,
        metadata={
            "help": "Token weighting for KL loss. "
            "1.0 = disabled (all tokens equal weight). "
            "0.0 = Soft-OR as continuous weight. "
            "(0,1) = hard top-k selection, only retain this fraction of tokens by Soft-OR score. "
            "Only applies to KL loss; GRPO loss (kl_mode=-3 negative) is unaffected."
        },
    )
    use_token_masking: bool = field(
        default=True,
        metadata={
            "help": "Mask low-score tokens in the negative-sample GRPO loss using probability and entropy."
        },
    )
    rl_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for the GRPO RL branch in kl_mode=-3. Scales negative sample loss "
                    "to align magnitude with FKL positive sample loss."
        },
    )
    privilege_info: str = field(
        default="answer",
        metadata={
            "help": "What to use as the teacher's privileged information in teacher prompts. "
                    "'answer' (default): the voted pseudo-label answer string. "
                    "'trajectory': the full completion text of the shortest positive rollout. "
                    "'none': no privileged info (teacher sees the plain problem only)."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((TTPOScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_config = (
            f"ttpo_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}_"
            f"K{script_args.num_rollouts}"
        )

    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "max_tokens": script_args.max_tokens,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "num_rollouts": script_args.num_rollouts,
                "num_train_rollouts": script_args.num_train_rollouts,
                "min_consensus_count": script_args.min_consensus_count,
                "use_gt": script_args.use_gt,
                "jsd_token_clip": script_args.jsd_token_clip,
                "jsd_clip_mode": script_args.jsd_clip_mode,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay,
                "fixed_teacher": script_args.fixed_teacher,
                "periodic_sync_teacher": script_args.periodic_sync_teacher,
                "teacher_sync_interval": script_args.teacher_sync_interval,
                "student_thinking": script_args.student_thinking,
                "teacher_thinking": script_args.teacher_thinking,
                "kl_mode": script_args.kl_mode,
                "positive_fraction": script_args.positive_fraction,
                "pos_select": script_args.pos_select,
                "neg_select": script_args.neg_select,
                "token_weighting_retention": script_args.token_weighting_retention,
                "privilege_info": script_args.privilege_info,
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

    dataset = load_dataset(script_args.dataset_name)
    train_dataset = dataset["train"]
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    training_args.presence_penalty = script_args.presence_penalty

    trainer = TTPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        num_rollouts=script_args.num_rollouts,
        num_train_rollouts=script_args.num_train_rollouts,
        min_consensus_count=script_args.min_consensus_count,
        use_gt=script_args.use_gt,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        jsd_clip_mode=script_args.jsd_clip_mode,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        fixed_teacher=script_args.fixed_teacher,
        periodic_sync_teacher=script_args.periodic_sync_teacher,
        teacher_sync_interval=script_args.teacher_sync_interval,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        kl_mode=script_args.kl_mode,
        max_tokens=script_args.max_tokens,
        positive_fraction=script_args.positive_fraction,
        pos_select=script_args.pos_select,
        neg_select=script_args.neg_select,
        token_weighting_retention=script_args.token_weighting_retention,
        use_token_masking=script_args.use_token_masking,
        rl_weight=script_args.rl_weight,
        privilege_info=script_args.privilege_info,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

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
