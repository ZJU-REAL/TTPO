import os
import wandb
import re

from math_verify import parse, verify

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


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with GRPO-specific options."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name) to log runs under."},
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "WandB project name to log runs under."},
    )


def extract_boxed_answer(text):
    """
    Extract the answer from \\boxed{} format.
    For thinking models, only searches after </think> to avoid picking up
    intermediate answers from the thinking block.
    Handles nested braces correctly (e.g. \\boxed{\\frac{1}{2}}).
    """
    # For thinking models (e.g. Qwen3), only look after </think>
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text

    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None


def _preprocess_for_parse(answer):
    """Convert ratio notation a:b → \\frac{a}{b} so math_verify can parse it."""
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer


def reward_correctness(completions, Answer, **kwargs):
    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, Answer)):
        pred_answer = extract_boxed_answer(completion)

        reward = 0.0

        # Try math_verify for mathematical equivalence (handles fractions, algebra, etc.)
        # Only use it when both sides actually parse to something (avoids silent None returns
        # for MCQ answers like "E" which parse() returns None for)
        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass

        # Fallback: whitespace-stripped string match (handles MCQ like "E", "A", etc.)
        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0

        rewards.append(reward)

    return rewards


def make_format_prompt(tokenizer):
    """
    Returns a formatting function that applies the tokenizer's chat template.
    """

    def format_prompt(example):
        messages = [
            {
                "role": "user",
                "content": f"Problem: {example['Question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "Answer": example["Answer"]}

    return format_prompt


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-5 -> "2e-5")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_name = (
            f"GRPO_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Prompt Length: {training_args.max_prompt_length}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
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
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
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

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    dataset = load_dataset("siyanzhao/Openthoughts_math_30k_opsd")
    train_dataset = dataset["train"]

    # Apply the format_prompt function to create the expected structure
    format_prompt = make_format_prompt(tokenizer)
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Auto-resume from latest checkpoint if one exists
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

    # Save model
    trainer.save_model(training_args.output_dir)
