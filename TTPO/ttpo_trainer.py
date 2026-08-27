import os
import random
import time
from collections import deque, defaultdict
from typing import Callable, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
)
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerControl
from trl import SFTTrainer
from trl.extras.profiling import profiling_decorator
from transformers.utils import is_peft_available
from trl.import_utils import is_vllm_available
from trl.experimental.gold.gold_config import GOLDConfig

if is_peft_available():
    from peft import PeftConfig, get_peft_model

from trl.trainer.utils import DataCollatorForChatML as DataCollator, ensure_master_addr_port, disable_dropout_in_model, empty_cache
from opsd_trainer import OPSDTrainer, EMAUpdateCallback
from ttpo_data_collator import TTPODataCollator
from ttpo_voting import extract_boxed_answer, majority_vote

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from trl.extras.vllm_client import VLLMClient

from accelerate.utils import gather_object, is_peft_model
from accelerate.state import PartialState
from torch.distributed import ReduceOp
from contextlib import nullcontext
from trl.models.utils import unwrap_model_for_generation


class PeriodicSyncCallback(TrainerCallback):
    """Hard-copy student weights into the teacher snapshot every N optimizer steps."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            self.trainer.periodic_sync_teacher
            and self.trainer.accelerator.sync_gradients
            and state.global_step % self.trainer.teacher_sync_interval == 0
        ):
            self.trainer._sync_teacher_snapshot()


class TTPOVLLMSyncCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class StepTimerCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self._last_end = None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        now = time.perf_counter()
        if self._last_end is not None:
            self.trainer._metrics["train"]["time/step_sec"].append(now - self._last_end)
        self._last_end = now


class TTPOTrainer(SFTTrainer):
    _tag_names = ["trl", "ttpo"]
    _name = "TTPO"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        num_rollouts: int = 8,
        num_train_rollouts: int | None = None,
        min_consensus_count: int = 2,
        use_gt: bool = False,
        jsd_token_clip: float | None = None,
        jsd_clip_mode: str = "element",
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        fixed_teacher: bool = False,
        periodic_sync_teacher: bool = False,
        teacher_sync_interval: int = 10,
        student_thinking: bool = False,
        teacher_thinking: bool = True,
        kl_mode: float = -2,
        rl_weight: float = 0.01,
        max_tokens: int | None = None,
        positive_fraction: float = -1,
        pos_select: int = 0,
        neg_select: int = 0,
        token_weighting_retention: float = 1.0,
        use_token_masking: bool = False,
        privilege_info: str = "answer",
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        if data_collator is None:
            data_collator = TTPODataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                student_thinking=student_thinking,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.num_rollouts = num_rollouts  # K_gen
        self.num_train_rollouts = num_train_rollouts if num_train_rollouts is not None else num_rollouts
        if self.num_train_rollouts > self.num_rollouts:
            raise ValueError(
                f"num_train_rollouts ({self.num_train_rollouts}) must be <= num_rollouts ({self.num_rollouts})."
            )
        self.min_consensus_count = min_consensus_count
        self.use_gt = use_gt
        self.jsd_token_clip = jsd_token_clip
        self.jsd_clip_mode = jsd_clip_mode

        # Teacher source / anchor for the answer-conditioned teacher forward pass.
        # The teacher is the same model conditioned on a prefix containing the voted
        # pseudo-label (analogous to naive OPSD). Anchor options:
        #   - dynamic (default): live student weights (no extra bookkeeping).
        #   - use_ema_teacher:   slow EMA snapshot of the student.
        #   - fixed_teacher:     frozen base model (LoRA adapters disabled).
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None  # lazily initialized on first optimizer step
        self.fixed_teacher = fixed_teacher
        self.periodic_sync_teacher = periodic_sync_teacher
        self.teacher_sync_interval = teacher_sync_interval
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.kl_mode = kl_mode
        self.rl_weight = rl_weight
        self.max_tokens = max_tokens if max_tokens is not None else args.max_completion_length
        self.positive_fraction = positive_fraction
        self.pos_select = pos_select
        self.neg_select = neg_select
        self.token_weighting_retention = token_weighting_retention
        self.use_token_masking = use_token_masking
        if privilege_info not in ("answer", "trajectory", "none"):
            raise ValueError(
                f"privilege_info must be one of 'answer', 'trajectory', 'none', got {privilege_info!r}."
            )
        self.privilege_info = privilege_info

        # Shared parameters from GOLDConfig
        self.beta = args.beta
        self.temperature = args.temperature

        print(f"\n{'='*80}")
        print("TTPO (TTRL + Self-Distillation, all-rollout contrastive) TRAINER")
        print(f"  sampling temperature: {self.temperature}")
        print(f"  max_completion_length (sampling): {args.max_completion_length}")
        print(f"  max_tokens (training update): {self.max_tokens}")
        if self.kl_mode == -3:
            kl_mode_desc = f"pos=fkl, neg=grpo_rl (rl_weight={self.rl_weight})"
        elif self.kl_mode == -4:
            kl_mode_desc = f"pos=grpo_rl, neg=fkl (rl_weight={self.rl_weight})"
        elif self.kl_mode == -2:
            kl_mode_desc = "pos=rkl, neg=fkl"
        elif self.kl_mode == -1:
            kl_mode_desc = "pos=fkl, neg=rkl"
        elif 0 <= self.kl_mode <= 1:
            kl_mode_desc = f"generalized JSD (beta={self.kl_mode})"
        else:
            kl_mode_desc = f"unknown (kl_mode={self.kl_mode})"
        print(f"  kl_mode: {self.kl_mode}  ->  {kl_mode_desc}")
        print(f"  jsd_token_clip: {self.jsd_token_clip}")
        print(f"  jsd_clip_mode: {self.jsd_clip_mode}")
        print(f"  token_weighting_retention: {self.token_weighting_retention}")
        print(f"  use_token_masking: {self.use_token_masking}")
        print(f"  privilege_info: {self.privilege_info}")
        print(f"  num_rollouts (K_gen): {self.num_rollouts}")
        print(f"  num_train_rollouts (K_train): {self.num_train_rollouts}")
        print(f"  positive_fraction: {self.positive_fraction}")
        print(f"  pos_select: {self.pos_select}")
        print(f"  neg_select: {self.neg_select}")
        print(f"  min_consensus_count: {self.min_consensus_count}")
        print(f"  use_gt: {self.use_gt} (pseudo-label = ground truth, min_consensus_count bypassed)")
        anchor = (
            "fixed (base, LoRA disabled)" if self.fixed_teacher else
            "ema snapshot" if self.use_ema_teacher else
            f"periodic sync (every {self.teacher_sync_interval} steps)" if self.periodic_sync_teacher else
            "dynamic (live student)"
        )
        print(f"  teacher = same model conditioned on consensus-answer prefix; anchor = {anchor}")
        if self.use_ema_teacher:
            print(f"  ema_decay: {self.ema_decay}")
        print(f"{'='*80}\n")

        teacher_flags = sum([self.fixed_teacher, self.use_ema_teacher, self.periodic_sync_teacher])
        if teacher_flags > 1:
            raise ValueError(
                "fixed_teacher, use_ema_teacher, and periodic_sync_teacher are mutually exclusive "
                "teacher strategies — set at most one."
            )
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during the voting forward pass."
            )

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED (TTPO)")
            print(f"EMA decay: {self.ema_decay}")
            print("Answer-conditioned teacher forward runs under an exponential moving")
            print("average of the student weights — a stable anchor that lags the live student.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED (TTPO)")
            print("Answer-conditioned teacher forward runs with LoRA adapters disabled")
            print("(base model = frozen initial policy). Strongest protection against drift.")
            print(f"{'='*80}\n")

        if self.periodic_sync_teacher:
            self.add_callback(PeriodicSyncCallback(self))
            print(f"\n{'='*80}")
            print("PERIODIC SYNC TEACHER MODE ENABLED (TTPO)")
            print(f"Teacher snapshot hard-copied from student every {self.teacher_sync_interval} steps.")
            print("Between syncs, teacher weights are frozen.")
            print(f"{'='*80}\n")

        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.add_callback(StepTimerCallback(self))
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        maxlen = (
            self.accelerator.num_processes
            * args.per_device_train_batch_size
            * args.steps_per_generation
            * self.num_rollouts
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rollout_idx": deque(maxlen=maxlen),
            "answer": deque(maxlen=maxlen),
            "is_selected": deque(maxlen=maxlen),
            "is_majority": deque(maxlen=maxlen),
            "direction_sign": deque(maxlen=maxlen),
            "ground_truth": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )
                if self.vllm_tensor_parallel_size > 1:
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()
                self.vllm_engine = LLM(
                    model=self.model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.num_rollouts,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )
                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1
            self.add_callback(TTPOVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = ["problem", "Answer", "answer"]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _generate_grouped_vllm(self, prompts_text):
        """Generate K trajectories for each given prompt text.

        In DDP mode each GPU independently generates all K rollouts for its own
        prompts using the local vLLM engine.

        Args:
            prompts_text: list of P prompt strings (already chat-templated, no padding tokens).

        Returns:
            completion_token_ids:   list of K*P token-id lists (raw, unpadded)
            completion_texts:       list of K*P decoded completion strings
            completion_logprobs:    list of K*P per-token logprob lists, or None if not requested
        """
        import time

        if not self.use_vllm:
            raise ValueError("_generate_grouped_vllm requires use_vllm=True")

        K = self.num_rollouts
        P = len(prompts_text)

        max_completion_length = self.generation_config.max_new_tokens
        temperature = self.generation_config.temperature
        top_k = self.generation_config.top_k if self.generation_config.top_k and self.generation_config.top_k > 0 else -1
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        start_time = time.time()
        need_logprobs = self.pos_select == 2 or self.neg_select == 2
        sampling_params = SamplingParams(
            n=K,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_completion_length,
            presence_penalty=presence_penalty,
            logprobs=0 if need_logprobs else None,
        )

        if self.vllm_mode == "colocate":
            all_outputs = self.vllm_engine.generate(
                prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        elif self.vllm_mode == "server":
            raise NotImplementedError(
                "Server mode generation not yet implemented for TTPO. "
                "Use vllm_mode='colocate'."
            )
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        elapsed_time = time.time() - start_time
        flat_outputs = [out for outputs in all_outputs for out in outputs.outputs]
        total_tokens = sum(len(out.token_ids) for out in flat_outputs)

        completion_token_ids = [list(out.token_ids) for out in flat_outputs]
        completion_texts = [
            self.processing_class.decode(out.token_ids, skip_special_tokens=False)
            for out in flat_outputs
        ]

        if need_logprobs:
            completion_logprobs = [
                [next(iter(lp.values())).logprob for lp in out.logprobs]
                for out in flat_outputs
            ]
        else:
            completion_logprobs = None

        print(
            f"vLLM grouped generation done - elapsed: {elapsed_time:.2f}s, prompts: {P}, K: {K}, "
            f"total trajectories: {P*K}, total tokens: {total_tokens}, "
            f"speed: {total_tokens/max(elapsed_time, 1e-6):.1f} tok/s"
        )

        return completion_token_ids, completion_texts, completion_logprobs

    def _generate_continuation_vllm(self, prefix_token_ids_list):
        """Generate short continuations from token-id prefixes to force answer extraction.

        Each entry in prefix_token_ids_list is: prompt_tokens + completion_tokens + suffix_tokens.
        Uses greedy decoding, stops at "}", returns list of decoded continuation strings.
        """
        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=64,
            stop=["}"],
            include_stop_str_in_output=True,
        )
        prompts = [{"prompt_token_ids": ids} for ids in prefix_token_ids_list]

        if self.vllm_mode == "colocate":
            outputs = self.vllm_engine.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        elif self.vllm_mode == "server":
            raise NotImplementedError(
                "Server mode continuation not yet implemented for TTPO. "
                "Use vllm_mode='colocate'."
            )
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        return [out.outputs[0].text for out in outputs]

    # Reuse OPSD generation and vLLM synchronization.

    generate_on_policy_outputs = OPSDTrainer.generate_on_policy_outputs
    _generate_on_policy_outputs_vllm = OPSDTrainer._generate_on_policy_outputs_vllm
    _move_model_to_vllm = OPSDTrainer._move_model_to_vllm
    _sync_fsdp_params_to_vllm = OPSDTrainer._sync_fsdp_params_to_vllm
    _wake_vllm_if_needed = OPSDTrainer._wake_vllm_if_needed
    _save_generation_outputs = OPSDTrainer._save_generation_outputs

    # Reuse OPSD's optional EMA teacher anchor.

    _update_ema = OPSDTrainer._update_ema
    _ema_teacher_context = OPSDTrainer._ema_teacher_context

    def _sync_teacher_snapshot(self):
        """Hard-copy current student weights into the teacher snapshot.

        On first call, lazily initializes `_ema_params` as an exact copy (same as EMA init).
        On subsequent calls, overwrites the snapshot with the current student weights.
        Reuses `_ema_teacher_context` for the teacher forward pass.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nPeriodic-sync teacher initialized: {n_tensors} tensors, "
                        f"{n_params:,} parameters (sync every {self.teacher_sync_interval} steps)"
                    )
                    return

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.copy_(param.data)
        else:
            if self._ema_params is None:
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nPeriodic-sync teacher initialized: {n_tensors} tensors, "
                    f"{n_params:,} parameters (sync every {self.teacher_sync_interval} steps)"
                )
                return

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.copy_(param.data)

        print(f"[Step {self.state.global_step}] Teacher snapshot synced from student.")

    _teacher_transition_prompt = (
        "\n\nAfter reading the reference solution above, make sure you truly understand "
        "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
        "own words and independent reasoning, derive the same final answer to the problem above. "
        "Think step by step, explore different approaches, and don't be afraid to backtrack "
        "or reconsider if something doesn't work out:\n"
    )

    def _build_teacher_prompts(self, problems, pseudo_labels):
        """Build answer-conditioned teacher prompts (one per problem).

        Mirrors the non-reason-first branch of SelfDistillationDataCollator: the pseudo-label
        voted from the K rollouts is injected as the "reference solution" so that the teacher
        forward sees a problem-with-answer prefix.

        Returns:
            teacher_prompt_ids:     [P, max_len] padded token IDs
            teacher_prompt_attn:    [P, max_len] attention mask
            teacher_prompt_lengths: list[int] of length P, un-padded prompt lengths
            max_len:                int, padded prompt length
        """
        device = self.accelerator.device
        teacher_texts = []
        for problem, pseudo_label in zip(problems, pseudo_labels):
            if self.privilege_info == "none":
                user_message = (
                    f"Problem: {problem}\n\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}."
                )
            else:
                ref_solution = pseudo_label if pseudo_label is not None else ""
                user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a reference solution to this problem:\n"
                    f"=== Reference Solution Begin ===\n{ref_solution}\n=== Reference Solution End ===\n"
                    f"{self._teacher_transition_prompt}\n"
                    f"Please reason step by step, and put your final answer within \\boxed{{}}."
                )
            messages = [{"role": "user", "content": user_message}]
            teacher_texts.append(
                self.processing_class.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
                )
            )

        prompt_max_length = (
            max(1, self.args.max_length - self.max_tokens) if self.args.max_length else None
        )
        encoded_no_pad = self.processing_class(
            teacher_texts,
            padding=False,
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        )
        teacher_prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_len = max(teacher_prompt_lengths)

        encoded = self.processing_class(
            teacher_texts,
            padding="max_length",
            truncation=True if prompt_max_length else False,
            max_length=max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        teacher_prompt_ids = encoded["input_ids"].to(device)
        teacher_prompt_attn = encoded["attention_mask"].to(device)
        return teacher_prompt_ids, teacher_prompt_attn, teacher_prompt_lengths, max_len

    @torch.no_grad()
    def _compute_selection_kl_scores(self, prompts_text, completion_token_ids, pseudo_labels, problems, P, K_gen, device):
        """Compute mean per-token KL(teacher || student) for all K_gen*P trajectories.

        Used by select=1 strategy. Does not participate in training loss.
        """
        pad_token_id = self.processing_class.pad_token_id
        max_completion_length = self.max_tokens

        # Encode student prompts
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        student_encoded_no_pad = self.processing_class(
            prompts_text,
            padding=False,
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)
        student_encoded = self.processing_class(
            prompts_text,
            padding="max_length",
            truncation=True if prompt_max_length else False,
            max_length=max_student_prompt_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        student_prompt_ids = student_encoded["input_ids"].to(device)  # [P, S]

        # Build teacher prompts
        teacher_prompt_ids, _, teacher_prompt_lengths, max_teacher_prompt_len = (
            self._build_teacher_prompts(problems, pseudo_labels)
        )

        # Pad all K_gen*P completions
        padded_completions = []
        for gidx in range(K_gen * P):
            comp_ids = completion_token_ids[gidx]
            comp_tensor = torch.tensor(comp_ids[:max_completion_length], device=device, dtype=student_prompt_ids.dtype)
            if comp_tensor.numel() < max_completion_length:
                pad_len = max_completion_length - comp_tensor.numel()
                comp_tensor = torch.cat([comp_tensor, comp_tensor.new_full((pad_len,), pad_token_id)])
            padded_completions.append(comp_tensor)
        padded_completions = torch.stack(padded_completions)  # [K_gen*P, T]

        # Build full sequences
        student_prompt_rep = student_prompt_ids.repeat_interleave(K_gen, dim=0)  # [K_gen*P, S]
        student_input_ids = torch.cat([student_prompt_rep, padded_completions], dim=1)
        student_attn = torch.ones_like(student_input_ids)
        if pad_token_id is not None:
            student_attn[student_input_ids == pad_token_id] = 0

        teacher_prompt_rep = teacher_prompt_ids.repeat_interleave(K_gen, dim=0)
        teacher_input_ids = torch.cat([teacher_prompt_rep, padded_completions], dim=1)
        teacher_attn = torch.ones_like(teacher_input_ids)
        if pad_token_id is not None:
            teacher_attn[teacher_input_ids == pad_token_id] = 0

        # Per-sample prompt lengths repeated
        s_prompt_lens = [student_prompt_lengths[p] for p in range(P) for _ in range(K_gen)]
        t_prompt_lens = [teacher_prompt_lengths[p] for p in range(P) for _ in range(K_gen)]

        # Determine adapter context for teacher
        model = self.model

        def _make_teacher_context():
            if self.fixed_teacher and is_peft_model(model):
                return self.accelerator.unwrap_model(model).disable_adapter()
            elif (self.use_ema_teacher or self.periodic_sync_teacher) and self._ema_params is not None:
                return self._ema_teacher_context(model)
            else:
                return nullcontext()

        chunk_size = max(1, self.args.per_device_train_batch_size)
        scores = []

        for start in range(0, K_gen * P, chunk_size):
            end = min(start + chunk_size, K_gen * P)

            # Student forward
            s_out = model(input_ids=student_input_ids[start:end], attention_mask=student_attn[start:end])
            s_logits = s_out.logits
            del s_out

            # Teacher forward
            with _make_teacher_context():
                t_out = model(input_ids=teacher_input_ids[start:end], attention_mask=teacher_attn[start:end])
                t_logits = t_out.logits
                del t_out

            # Compute per-sample mean KL over response tokens
            for i in range(end - start):
                gidx = start + i
                s_plen = s_prompt_lens[gidx]
                t_plen = t_prompt_lens[gidx]
                actual_len = min(len(completion_token_ids[gidx]), max_completion_length)

                if actual_len == 0:
                    scores.append(float('-inf'))
                    continue

                # Response logits: predict tokens at positions [prompt_len, prompt_len + actual_len - 1]
                s_resp = s_logits[i, s_plen - 1: s_plen - 1 + actual_len, :]
                t_resp = t_logits[i, t_plen - 1: t_plen - 1 + actual_len, :]

                t_log_probs = F.log_softmax(t_resp / self.temperature, dim=-1)
                s_log_probs = F.log_softmax(s_resp / self.temperature, dim=-1)
                t_probs = torch.exp(t_log_probs)
                kl_per_token = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)  # [actual_len]
                scores.append(kl_per_token.mean().item())

            del s_logits, t_logits
            empty_cache()

        return scores

    def _normalize_token_scores(self, values, mask, clip_quantile=0.98):
        """Min-max normalize token scores after percentile clipping."""
        valid = values[mask.bool()]
        if valid.numel() == 0:
            return torch.zeros_like(values)
        valid_f = valid.float()
        clip_val = torch.quantile(valid_f, clip_quantile).to(values.dtype)
        v_min = valid.min()
        if (clip_val - v_min).float() < 1e-8:
            return torch.zeros_like(values)
        clipped = values.clamp(max=clip_val)
        return ((clipped - v_min) / (clip_val - v_min)).clamp(0, 1)

    def _compute_token_weights(self, student_log_probs, teacher_log_probs, mask):
        """Compute Soft-OR token weights from entropy and teacher-student divergence.

        Args:
            student_log_probs: [B, T, V] temperature-scaled student log probabilities
            teacher_log_probs: [B, T, V] temperature-scaled teacher log probabilities
            mask: [B, T] valid token mask

        Returns:
            soft_or: [B, T] token weight in [0, 1]
        """
        with torch.no_grad():
            student_probs = torch.exp(student_log_probs)
            h_t = -(student_probs * student_log_probs).sum(dim=-1)

            teacher_probs = torch.exp(teacher_log_probs)
            delta_t = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
            delta_t = delta_t.clamp(min=0)

            h_hat = self._normalize_token_scores(h_t, mask)
            delta_hat = self._normalize_token_scores(delta_t, mask)

            soft_or = h_hat + delta_hat - h_hat * delta_hat

        return soft_or

    def _apply_token_weighting(self, per_token_kl, student_log_probs, teacher_log_probs, mask):
        """Apply entropy/divergence-based weights to per-token KL loss.

        token_weighting_retention=1.0: use the standard mean over valid tokens.
        token_weighting_retention=0.0: use continuous Soft-OR weights.
        Values in (0,1): retain the corresponding top-scoring token fraction.
        """
        if self.token_weighting_retention >= 1.0:
            per_token_kl = per_token_kl * mask
            token_counts = mask.sum(dim=-1).clamp(min=1.0)
            return per_token_kl.sum(dim=-1) / token_counts

        soft_or = self._compute_token_weights(student_log_probs, teacher_log_probs, mask.bool())

        if self.token_weighting_retention == 0.0:
            per_token_kl = per_token_kl * mask * soft_or
            token_counts = mask.sum(dim=-1).clamp(min=1.0)
            per_sample = per_token_kl.sum(dim=-1) / token_counts
        else:
            num_valid = mask.sum(dim=-1)
            k_per_sample = (num_valid * self.token_weighting_retention).long().clamp(min=1)

            scores = soft_or.clone()
            scores[~mask.bool()] = -float('inf')

            max_k = k_per_sample.max().item()
            _, topk_indices = scores.topk(max_k, dim=-1)

            arange = torch.arange(max_k, device=mask.device).unsqueeze(0)
            selection = (arange < k_per_sample.unsqueeze(-1)).float()

            selected_kl = per_token_kl.gather(-1, topk_indices) * selection
            per_sample = selected_kl.sum(dim=-1) / k_per_sample.float()

        with torch.no_grad():
            valid_scores = soft_or[mask.bool()]
            local_q4_sum = (valid_scores < 0.1).float().sum().item() if valid_scores.numel() > 0 else 0.0
            local_valid_count = float(valid_scores.numel())
            local_weight_sum = valid_scores.sum().item() if valid_scores.numel() > 0 else 0.0
            local_sel_kl_sum = local_sel_kl_count = 0.0
            local_drop_kl_sum = local_drop_kl_count = 0.0
            if self.token_weighting_retention != 0.0:
                masked_kl = per_token_kl * mask
                selected_mask = torch.zeros_like(mask, dtype=selection.dtype)
                selected_mask.scatter_(-1, topk_indices, selection)
                dropped_mask = mask - selected_mask
                local_sel_kl_sum = (masked_kl * selected_mask).sum().item()
                local_sel_kl_count = float(selected_mask.sum().item())
                local_drop_kl_sum = (masked_kl * dropped_mask).sum().item()
                local_drop_kl_count = float(dropped_mask.sum().item())
        # Collective op: must run unconditionally on every rank (no rank-divergent branch).
        (
            g_q4_sum, g_valid_count, g_weight_sum,
            g_sel_kl_sum, g_sel_kl_count, g_drop_kl_sum, g_drop_kl_count,
        ) = self._dist_reduce(
            [
                local_q4_sum, local_valid_count, local_weight_sum,
                local_sel_kl_sum, local_sel_kl_count, local_drop_kl_sum, local_drop_kl_count,
            ],
            op=ReduceOp.SUM,
        )
        if g_valid_count > 0:
            self._metrics["train"]["token_weighting/q4_frac"].append(g_q4_sum / g_valid_count)
        if self.token_weighting_retention == 0.0:
            if g_valid_count > 0:
                self._metrics["train"]["token_weighting/mean_weight"].append(g_weight_sum / g_valid_count)
        else:
            if g_sel_kl_count > 0:
                self._metrics["train"]["token_weighting/selected_kl_mean"].append(g_sel_kl_sum / g_sel_kl_count)
            if g_drop_kl_count > 0:
                self._metrics["train"]["token_weighting/dropped_kl_mean"].append(g_drop_kl_sum / g_drop_kl_count)

        return per_sample

    def weighted_jsd_loss(
        self,
        student_logits,
        teacher_log_probs,
        labels,
        direction_signs,
        token_clip=None,
        clip_mode="element",
    ):
        """Direction-aware KL / generalized JSD loss.

        kl_mode in [0, 1]: generalized JSD applied uniformly to all samples.
            0 = forward KL, 1 = reverse KL, (0,1) = JSD with mixture distribution.
        kl_mode == -1: positive samples use forward KL, negative use reverse KL.
        kl_mode == -2: positive samples use reverse KL, negative use forward KL.
        """
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)

        if 0 <= self.kl_mode <= 1:
            # Generalized JSD (OPSD-compatible), all samples treated uniformly
            beta = self.kl_mode
            if beta == 0:
                per_element = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
            elif beta == 1:
                per_element = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
            else:
                beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
                mixture_log_probs = torch.logsumexp(
                    torch.stack([student_log_probs + torch.log1p(-beta_t),
                                 teacher_log_probs + torch.log(beta_t)]),
                    dim=0,
                )
                kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
                kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
                per_element = beta_t * kl_teacher + (1 - beta_t) * kl_student

            if token_clip is not None and clip_mode == "element":
                per_element = per_element.clamp(max=token_clip)
            per_token_kl = per_element.sum(dim=-1)  # [B, T]
            del per_element
            if token_clip is not None and clip_mode == "token":
                per_token_kl = per_token_kl.clamp(max=token_clip)
        elif self.kl_mode in (-1, -2):
            # Direction-aware KL: different KL for positive vs negative samples
            kl_fkl_raw = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_rkl_raw = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)

            if token_clip is not None and clip_mode == "element":
                kl_rkl_raw = kl_rkl_raw.clamp(max=token_clip)
                kl_fkl_raw = kl_fkl_raw.clamp(max=token_clip)

            kl_rkl = kl_rkl_raw.sum(dim=-1)  # [B, T]
            kl_fkl = kl_fkl_raw.sum(dim=-1)  # [B, T]
            del kl_rkl_raw, kl_fkl_raw

            is_positive = (direction_signs > 0).to(student_log_probs.dtype).unsqueeze(-1)  # [B, 1]
            if self.kl_mode == -1:    # positive fkl, negative rkl
                per_token_kl = is_positive * kl_fkl + (1 - is_positive) * kl_rkl
            else:                     # -2: positive rkl, negative fkl
                per_token_kl = is_positive * kl_rkl + (1 - is_positive) * kl_fkl

            if token_clip is not None and clip_mode == "token":
                per_token_kl = per_token_kl.clamp(max=token_clip)
        else:
            raise ValueError(
                f"Unsupported kl_mode={self.kl_mode}. "
                "Use [0,1] for generalized JSD, -1 (pos fkl / neg rkl), -2 (pos rkl / neg fkl)."
            )

        mask = (labels != -100).to(per_token_kl.dtype)
        per_sample = self._apply_token_weighting(per_token_kl, student_log_probs, teacher_log_probs, mask)

        return per_sample.mean()

    def _compute_hybrid_fkl_grpo_loss(
        self, student_logits, teacher_log_probs, labels, direction_signs, advantages,
    ):
        """Hybrid loss: FKL for positive samples, GRPO policy gradient for negative samples."""
        mask = (labels != -100).float()
        token_counts = mask.sum(dim=-1).clamp(min=1.0)
        pos_mask = direction_signs > 0  # [B]
        neg_mask = direction_signs < 0  # [B]

        # Positive: Forward KL (temperature-scaled)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        fkl = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        if self.jsd_token_clip is not None and self.jsd_clip_mode == "element":
            fkl = fkl.clamp(max=self.jsd_token_clip)
        fkl_per_token = fkl.sum(dim=-1)  # [B, T]
        if self.jsd_token_clip is not None and self.jsd_clip_mode == "token":
            fkl_per_token = fkl_per_token.clamp(max=self.jsd_token_clip)

        fkl_per_sample = self._apply_token_weighting(fkl_per_token, student_log_probs, teacher_log_probs, mask)

        # Negative: GRPO policy gradient
        log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)  # [B, T, V]
        token_ids = labels.clamp(min=0).unsqueeze(-1)  # [B, T, 1]
        per_token_logps = log_probs.gather(-1, token_ids).squeeze(-1)  # [B, T]
        rl_per_token = -advantages.unsqueeze(-1) * per_token_logps  # [B, T]

        if self.use_token_masking:
            with torch.no_grad():
                student_probs_full = torch.exp(log_probs).float()
                student_entropy = -(student_probs_full * log_probs.float()).nan_to_num(0.0).sum(dim=-1)  # [B, T]

                low_entropy_hat = 1.0 - self._normalize_token_scores(student_entropy, mask)

                score = -per_token_logps.float() * low_entropy_hat
                valid = score[mask.bool()]
                thresh = torch.quantile(valid, 0.5)
                token_mask = (score >= thresh).to(rl_per_token.dtype)

            effective_mask = mask * token_mask
            rl_per_sample = (rl_per_token * effective_mask).sum(dim=-1) / token_counts
        else:
            rl_per_sample = (rl_per_token * mask).sum(dim=-1) / token_counts  # [B]

        # Combine: all samples mean, rl_weight aligns GRPO magnitude to FKL
        if self.kl_mode == -3:
            # -3: pos=fkl, neg=grpo
            per_sample_loss = torch.where(pos_mask, fkl_per_sample, self.rl_weight * rl_per_sample)
        else:
            # -4: pos=grpo, neg=fkl
            per_sample_loss = torch.where(pos_mask, self.rl_weight * rl_per_sample, fkl_per_sample)

        # Log both components (unweighted, for monitoring magnitude).
        # Aggregate globally across processes so the logged mean reflects every rank's
        # samples, not just rank 0's local micro-batch. With 1 prompt/GPU, rank 0's
        # single prompt is frequently all-negative while other ranks carry the positives.
        with torch.no_grad():
            pos_source, neg_source = (
                (fkl_per_sample, rl_per_sample) if self.kl_mode == -3 else (rl_per_sample, fkl_per_sample)
            )
            local_pos_sum = pos_source[pos_mask].sum().item() if pos_mask.any() else 0.0
            local_pos_count = float(pos_mask.sum().item())
            local_neg_sum = neg_source[neg_mask].sum().item() if neg_mask.any() else 0.0
            local_neg_count = float(neg_mask.sum().item())
            local_token_mask_sel_sum = local_token_mask_sel_count = 0.0
            local_token_mask_all_sum = local_token_mask_all_count = 0.0
            if self.use_token_masking and neg_mask.any():
                neg_logps = per_token_logps[neg_mask]
                neg_valid = mask[neg_mask].bool()
                sel = (token_mask[neg_mask] * mask[neg_mask]).bool()
                if sel.any():
                    local_token_mask_sel_sum = neg_logps[sel].float().sum().item()
                    local_token_mask_sel_count = float(sel.sum().item())
                if neg_valid.any():
                    local_token_mask_all_sum = neg_logps[neg_valid].float().sum().item()
                    local_token_mask_all_count = float(neg_valid.sum().item())
        # Collective op: must run unconditionally on every rank (no rank-divergent branch).
        (
            g_pos_sum, g_pos_count, g_neg_sum, g_neg_count,
            g_token_mask_sel_sum, g_token_mask_sel_count, g_token_mask_all_sum, g_token_mask_all_count,
        ) = self._dist_reduce(
            [
                local_pos_sum, local_pos_count, local_neg_sum, local_neg_count,
                local_token_mask_sel_sum, local_token_mask_sel_count,
                local_token_mask_all_sum, local_token_mask_all_count,
            ],
            op=ReduceOp.SUM,
        )
        pos_mean = g_pos_sum / g_pos_count if g_pos_count > 0 else 0.0
        neg_mean = g_neg_sum / g_neg_count if g_neg_count > 0 else 0.0
        self._metrics["train"]["loss/pos"].append(pos_mean)
        self._metrics["train"]["loss/neg"].append(neg_mean)
        if self.use_token_masking:
            if g_token_mask_sel_count > 0:
                self._metrics["train"]["token_masking/selected_logp_mean"].append(g_token_mask_sel_sum / g_token_mask_sel_count)
            if g_token_mask_all_count > 0:
                self._metrics["train"]["token_masking/all_logp_mean"].append(g_token_mask_all_sum / g_token_mask_all_count)

        return per_sample_loss.mean()

    def _dist_reduce(self, values, op=ReduceOp.SUM):
        """All-reduce a list of scalars across processes and return the reduced python list.

        No-op when not running distributed. MUST be called unconditionally on every rank
        (it is a collective op), so callers must not place it behind rank-divergent branches.
        """
        import torch.distributed as dist

        vec = torch.tensor(values, dtype=torch.float64, device=self.accelerator.device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(vec, op=op)
        return vec.tolist()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        student_input_ids = inputs["student_input_ids"]
        student_attn = inputs["student_attention_mask"]
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attn = inputs["teacher_attention_mask"]
        direction_signs = inputs["direction_signs"]  # [B], values in {+1, -1}

        shifted_labels = inputs["labels"][:, student_prompt_len:]
        labels_mask = shifted_labels != -100  # [B, T_gen]

        # Student forward (live, with grad)
        outputs = model(input_ids=student_input_ids, attention_mask=student_attn)
        student_logits = outputs.logits[:, student_prompt_len - 1: -1, :]  # [B, T_gen, V]
        del outputs
        empty_cache()

        # Teacher forward: same model conditioned on the answer-bearing prefix.
        # Anchor: dynamic (default) / EMA snapshot / fixed base (LoRA disabled).
        if self.fixed_teacher and is_peft_model(model):
            adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
        elif (self.use_ema_teacher or self.periodic_sync_teacher) and self._ema_params is not None:
            adapter_context = self._ema_teacher_context(model)
        else:
            adapter_context = nullcontext()

        with torch.no_grad(), adapter_context:
            teacher_outputs = model(input_ids=teacher_input_ids, attention_mask=teacher_attn)
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_len - 1: -1, :].detach()
            del teacher_outputs
            empty_cache()

        teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)

        del teacher_logits

        # Monitoring metrics (token-weighted, aggregated across processes)
        with torch.no_grad():
            if labels_mask.any():
                student_probs = F.softmax(student_logits.detach().float() / self.temperature, dim=-1)
                student_entropy = -(student_probs * torch.log(student_probs + 1e-10)).sum(-1)
                teacher_probs = torch.exp(teacher_log_probs)
                teacher_entropy = -(teacher_probs * teacher_log_probs).sum(-1)
                se = student_entropy[labels_mask]
                te = teacher_entropy[labels_mask]
                local_student_sum = se.sum().item()
                local_teacher_sum = te.sum().item()
                local_token_count = float(se.numel())
            else:
                local_student_sum = 0.0
                local_teacher_sum = 0.0
                local_token_count = 0.0

        g_student_sum, g_teacher_sum, g_token_count = self._dist_reduce(
            [local_student_sum, local_teacher_sum, local_token_count], op=ReduceOp.SUM
        )
        if g_token_count > 0:
            g_student_entropy = g_student_sum / g_token_count
            g_teacher_entropy = g_teacher_sum / g_token_count
            self._metrics["train"]["student_entropy"].append(g_student_entropy)
            self._metrics["train"]["teacher_entropy"].append(g_teacher_entropy)
            self._metrics["train"]["entropy_gap"].append(g_teacher_entropy - g_student_entropy)

        if self.kl_mode in (-3, -4):
            loss = self._compute_hybrid_fkl_grpo_loss(
                student_logits=student_logits,
                teacher_log_probs=teacher_log_probs,
                labels=shifted_labels,
                direction_signs=direction_signs,
                advantages=inputs["advantages"],
            )
        else:
            loss = self.weighted_jsd_loss(
                student_logits=student_logits,
                teacher_log_probs=teacher_log_probs,
                labels=shifted_labels,
                direction_signs=direction_signs,
                token_clip=self.jsd_token_clip,
                clip_mode=self.jsd_clip_mode,
            )

        del student_logits, teacher_log_probs
        empty_cache()

        if return_outputs:
            class MinimalOutput:
                pass
            out = MinimalOutput()
            out.loss = loss
            return (loss, out)
        return loss

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """Override HF Trainer's get_batch_samples for TTPO.

        DDP mode: each GPU independently processes its own prompts from the
        DistributedSampler. Constraint: (batch × grad_accum) % K_train == 0.

        Flow:
        1. Compute num_prompts = (batch × grad_accum) / K_train
        2. Pull num_prompts prompts from the dataloader
        3. Generate K_gen rollouts per prompt, vote, select K_train, build tensors
        4. Split into grad_accum micro-batches of size batch
        """
        K_train = self.num_train_rollouts
        K_gen = self.num_rollouts

        if K_gen <= 1:
            return super().get_batch_samples(epoch_iterator, num_batches, device)

        batch = self.args.per_device_train_batch_size
        total_items = batch * num_batches

        if total_items % K_train != 0:
            raise ValueError(
                f"per_device_train_batch_size × gradient_accumulation_steps "
                f"({batch} × {num_batches} = {total_items}) must be divisible by "
                f"num_train_rollouts ({K_train})."
            )
        num_prompts = total_items // K_train

        raw_batches = []
        prompts_collected = 0
        while prompts_collected < num_prompts:
            try:
                raw_batch = next(epoch_iterator)
            except StopIteration:
                break
            for k, v in list(raw_batch.items()):
                if torch.is_tensor(v):
                    raw_batch[k] = v.to(device)
            raw_batches.append(raw_batch)
            prompts_collected += len(raw_batch["problems"])

        all_items = self._generate_all_items(raw_batches, num_prompts, device)

        scalar_keys = {"student_prompt_length", "teacher_prompt_length"}
        micro_batches = []
        for i in range(0, total_items, batch):
            mb = {}
            for k, v in all_items.items():
                if k in scalar_keys:
                    mb[k] = v
                else:
                    mb[k] = v[i:i + batch]
            micro_batches.append(mb)

        return micro_batches, None

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        if self.num_rollouts == 1:
            return self._training_step_single(inputs, model, num_items_in_batch)
        # inputs is already a pre-processed chunk dict (one trajectory per prompt)
        return super().training_step(model, inputs, num_items_in_batch)

    def _generate_all_items(
        self, raw_batches: list[dict], total_prompts: int, device: torch.device
    ) -> dict[str, torch.Tensor | int]:
        """Generate rollouts, vote, select, and build tensors for total_prompts prompts.

        Returns a flat dict of tensors with shape [K_train * total_prompts, seq_len],
        plus scalar prompt lengths. In DDP mode each GPU calls this independently
        for its own prompts.
        """
        K_gen = self.num_rollouts
        K_train = self.num_train_rollouts

        if not self.use_vllm:
            raise NotImplementedError(
                "TTPO K_gen>1 path currently requires use_vllm=True."
            )

        # Merge the accumulated prompt batches.
        all_student_prompts = []
        all_problems = []
        all_gt_answers = []
        for rb in raw_batches:
            if "student_prompts" in rb and torch.is_tensor(rb["student_prompts"]):
                batch_prompts = self.processing_class.batch_decode(
                    rb["student_prompts"], skip_special_tokens=False
                )
                if self.processing_class.pad_token:
                    batch_prompts = [p.replace(self.processing_class.pad_token, "") for p in batch_prompts]
                all_student_prompts.extend(batch_prompts)
            all_problems.extend(rb.get("problems", []))
            all_gt_answers.extend(rb.get("answers", []))

        prompts_text = all_student_prompts[:total_prompts]
        problems = all_problems[:total_prompts]
        gt_answers = all_gt_answers[:total_prompts]
        P = len(prompts_text)

        self._wake_vllm_if_needed()
        completion_token_ids, completion_texts, completion_logprobs = self._generate_grouped_vllm(prompts_text)
        all_answers = [extract_boxed_answer(t) for t in completion_texts]

        pre_continuation_extractable = sum(1 for a in all_answers if a not in (None, ""))

        # Prefix-guided continuation for rollouts without extractable answers
        missing_indices = [i for i, a in enumerate(all_answers) if a in (None, "")]
        if missing_indices:
            prompt_encoded = self.processing_class(
                prompts_text, padding=False, add_special_tokens=False
            )
            think_end_str = "</think>"
            boxed_prefix_str = "\n\n\\boxed{"
            suffix_with_think_ids = self.processing_class.encode(
                think_end_str + boxed_prefix_str, add_special_tokens=False
            )
            suffix_without_think_ids = self.processing_class.encode(
                boxed_prefix_str, add_special_tokens=False
            )

            prefix_token_ids_list = []
            for idx in missing_indices:
                p_idx = idx // K_gen
                prompt_ids = prompt_encoded["input_ids"][p_idx]
                comp_ids = completion_token_ids[idx]
                if think_end_str in completion_texts[idx]:
                    suffix_ids = suffix_without_think_ids
                else:
                    suffix_ids = suffix_with_think_ids
                prefix_token_ids_list.append(prompt_ids + comp_ids + suffix_ids)

            self._wake_vllm_if_needed()
            continuations = self._generate_continuation_vllm(prefix_token_ids_list)

            for i, idx in enumerate(missing_indices):
                cont_text = continuations[i]
                answer = cont_text.rstrip("}").strip() if cont_text else None
                if answer:
                    all_answers[idx] = answer

            forced_count = sum(1 for i, idx in enumerate(missing_indices) if all_answers[idx] not in (None, ""))
            print(
                f"[TTPO Step {self.state.global_step}] Prefix-guided continuation: "
                f"{forced_count}/{len(missing_indices)} truncated rollouts recovered."
            )

        # Vote before selection so the requested positive fraction can use vote results.
        def comp_len(global_idx):
            return len(completion_token_ids[global_idx])

        from ttpo_voting import answers_equivalent

        pseudo_labels = []
        consensus_counts = []
        prompts_with_consensus = 0
        consensus_threshold = 0 if self.use_gt else self.min_consensus_count
        for p in range(P):
            base = p * K_gen
            group = [all_answers[base + k] for k in range(K_gen)]
            if self.use_gt:
                gt = gt_answers[p] if p < len(gt_answers) else None
                if not gt:
                    raise ValueError(
                        f"use_gt=True requires a ground-truth answer for every sample, but prompt "
                        f"{p} has none. Problem: {problems[p][:200]!r}"
                    )
                pseudo_label = gt
                consensus_count = sum(1 for a in group if a and answers_equivalent(a, gt))
            else:
                pseudo_label, _, consensus_count = majority_vote(group)
            pseudo_labels.append(pseudo_label if pseudo_label is not None else "")
            consensus_counts.append(consensus_count)
            if consensus_count >= consensus_threshold:
                prompts_with_consensus += 1

        # Compute the optional KL/log-probability selection scores.
        selection_scores_kl = None
        selection_scores_logp = None

        if 1 in (self.pos_select, self.neg_select):
            selection_scores_kl = self._compute_selection_kl_scores(
                prompts_text, completion_token_ids, pseudo_labels, problems, P, K_gen, device
            )

        if 2 in (self.pos_select, self.neg_select) and completion_logprobs is not None:
            selection_scores_logp = [
                sum(lps) / len(lps) if lps else float('-inf')
                for lps in completion_logprobs
            ]

        # Select K_train trajectories per prompt and assign update directions.
        selected_global_indices = []  # length K_train * P, indexes into [0, K_gen*P)
        selected_is_extractable = []  # length K_train * P, bool
        direction_signs_list = []     # length K_train * P, values in {+1, -1}
        backfill_count = 0

        def apply_select_strategy(ks, strategy, base):
            """Sort candidate indices by selection strategy. Returns sorted list."""
            if strategy == 0:
                return sorted(ks, key=lambda k: (all_answers[base + k] in (None, ""), comp_len(base + k)))
            elif strategy == 1:
                return sorted(ks, key=lambda k: (-selection_scores_kl[base + k], comp_len(base + k), k))
            elif strategy == 2:
                return sorted(ks, key=lambda k: (-selection_scores_logp[base + k], comp_len(base + k), k))
            elif strategy == 3:
                return sorted(ks, key=lambda k: (all_answers[base + k] in (None, ""), -comp_len(base + k)))
            return list(ks)

        for p in range(P):
            base = p * K_gen
            consensus_count = consensus_counts[p]
            pseudo_label = pseudo_labels[p]

            if self.positive_fraction == -2:
                # Naive selection: take first K_train rollouts, no quality filtering
                chosen_ks = list(range(K_train))
                selected_global_indices.extend(base + k for k in chosen_ks)
                selected_is_extractable.extend(
                    [all_answers[base + k] not in (None, "") for k in chosen_ks]
                )
                if consensus_count >= consensus_threshold and pseudo_label:
                    for k in chosen_ks:
                        ans = all_answers[base + k]
                        direction_signs_list.append(1 if (ans and answers_equivalent(ans, pseudo_label)) else -1)
                else:
                    ext_ks = [k for k in chosen_ks if all_answers[base + k] not in (None, "")]
                    best_k = min(ext_ks, key=lambda k: comp_len(base + k)) if ext_ks else None
                    for k in chosen_ks:
                        direction_signs_list.append(1 if k == best_k else -1)

            elif consensus_count >= consensus_threshold and pseudo_label:
                # Has consensus: split into pos/neg, determine ratio, select within each
                pos_ks = []
                neg_ks = []
                for k in range(K_gen):
                    ans = all_answers[base + k]
                    if ans and answers_equivalent(ans, pseudo_label):
                        pos_ks.append(k)
                    else:
                        neg_ks.append(k)
                N_pos, N_neg = len(pos_ks), len(neg_ks)

                if self.positive_fraction == -1:
                    # Dynamic sqrt-tempered ratio
                    if N_neg == 0:
                        K_pos_target, K_neg_target = min(K_train, N_pos), 0
                    elif N_pos == 0:
                        K_pos_target, K_neg_target = 0, min(K_train, N_neg)
                    else:
                        sqrt_pos, sqrt_neg = N_pos ** 0.5, N_neg ** 0.5
                        K_neg_star = K_train * sqrt_neg / (sqrt_pos + sqrt_neg)
                        K_neg_target = int(round(K_neg_star))
                        K_neg_target = max(1, min(K_neg_target, N_neg))
                        K_pos_target = K_train - K_neg_target
                        K_pos_target = max(1, min(K_pos_target, N_pos))
                        if K_pos_target + K_neg_target < K_train:
                            K_neg_target = min(K_train - K_pos_target, N_neg)
                        if K_pos_target + K_neg_target < K_train:
                            K_pos_target = min(K_train - K_neg_target, N_pos)
                else:
                    # Fixed ratio from positive_fraction
                    K_pos_target = round(K_train * self.positive_fraction)
                    K_neg_target = K_train - K_pos_target

                pos_ks = apply_select_strategy(pos_ks, self.pos_select, base)
                neg_ks = apply_select_strategy(neg_ks, self.neg_select, base)

                chosen_pos = pos_ks[:K_pos_target]
                chosen_neg = neg_ks[:K_neg_target]

                # Backfill if one category is insufficient
                total_chosen = len(chosen_pos) + len(chosen_neg)
                if total_chosen < K_train:
                    deficit = K_train - total_chosen
                    if len(chosen_pos) < K_pos_target:
                        extra_neg = neg_ks[len(chosen_neg):len(chosen_neg) + deficit]
                        chosen_neg.extend(extra_neg)
                    else:
                        extra_pos = pos_ks[len(chosen_pos):len(chosen_pos) + deficit]
                        chosen_pos.extend(extra_pos)

                chosen_ks = (chosen_pos + chosen_neg)[:K_train]
                selected_global_indices.extend(base + k for k in chosen_ks)
                selected_is_extractable.extend(
                    [all_answers[base + k] not in (None, "") for k in chosen_ks]
                )
                for k in chosen_ks:
                    ans = all_answers[base + k]
                    direction_signs_list.append(1 if (ans and answers_equivalent(ans, pseudo_label)) else -1)

            else:
                # No consensus fallback: select K_train using pos_select strategy, heuristic direction
                all_ks = apply_select_strategy(list(range(K_gen)), self.pos_select, base)
                chosen_ks = all_ks[:K_train]
                selected_global_indices.extend(base + k for k in chosen_ks)
                selected_is_extractable.extend(
                    [all_answers[base + k] not in (None, "") for k in chosen_ks]
                )
                backfill_count += sum(1 for k in chosen_ks if all_answers[base + k] in (None, ""))
                # Direction: shortest extractable = +1, rest = -1
                ext_in_chosen = [k for k in chosen_ks if all_answers[base + k] not in (None, "")]
                best_k = min(ext_in_chosen, key=lambda k: comp_len(base + k)) if ext_in_chosen else None
                for k in chosen_ks:
                    direction_signs_list.append(1 if k == best_k else -1)

        # Build student tensors for the selected trajectories.
        max_completion_length = self.max_tokens
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        student_encoded_no_pad = self.processing_class(
            prompts_text,
            padding=False,
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)
        student_encoded = self.processing_class(
            prompts_text,
            padding="max_length",
            truncation=True if prompt_max_length else False,
            max_length=max_student_prompt_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        student_prompt_ids = student_encoded["input_ids"].to(device)  # [P, S]

        pad_token_id = self.processing_class.pad_token_id
        student_prompt_ids_rep = student_prompt_ids.repeat_interleave(K_train, dim=0)  # [K_train*P, S]

        padded_completion_list = []
        for gidx in selected_global_indices:
            comp_ids = completion_token_ids[gidx]
            comp_tensor = torch.tensor(comp_ids, device=device, dtype=student_prompt_ids.dtype)
            if comp_tensor.numel() > max_completion_length:
                comp_tensor = comp_tensor[:max_completion_length]
            elif comp_tensor.numel() < max_completion_length:
                pad_len = max_completion_length - comp_tensor.numel()
                pad_t = torch.full((pad_len,), pad_token_id, device=device, dtype=comp_tensor.dtype)
                comp_tensor = torch.cat([comp_tensor, pad_t])
            padded_completion_list.append(comp_tensor)
        padded_completions = torch.stack(padded_completion_list)  # [K_train*P, T_gen]

        student_input_ids = torch.cat([student_prompt_ids_rep, padded_completions], dim=1)
        student_attention_mask = torch.ones_like(student_input_ids)
        if pad_token_id is not None:
            student_attention_mask[student_input_ids == pad_token_id] = 0

        student_prompt_lengths_tensor = torch.tensor(student_prompt_lengths, device=device)
        student_prompt_lengths_rep = student_prompt_lengths_tensor.repeat_interleave(K_train)

        labels = student_input_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = student_prompt_lengths_rep[i].item()
            labels[i, :actual_prompt_len] = -100
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Build answer-conditioned teacher tensors.
        if self.privilege_info == "trajectory":
            teacher_privileges = []
            for p in range(P):
                pos_indices = []
                for j in range(K_train):
                    flat_idx = p * K_train + j
                    if direction_signs_list[flat_idx] == 1:
                        pos_indices.append(selected_global_indices[flat_idx])
                if pos_indices:
                    shortest_pos_gidx = min(pos_indices, key=lambda gidx: len(completion_token_ids[gidx]))
                    teacher_privileges.append(completion_texts[shortest_pos_gidx])
                else:
                    teacher_privileges.append(pseudo_labels[p])
        else:
            teacher_privileges = pseudo_labels

        teacher_prompt_ids, teacher_prompt_attn, teacher_prompt_lengths, max_teacher_prompt_len = (
            self._build_teacher_prompts(problems, teacher_privileges)
        )
        teacher_prompt_ids_rep = teacher_prompt_ids.repeat_interleave(K_train, dim=0)
        teacher_input_ids = torch.cat([teacher_prompt_ids_rep, padded_completions], dim=1)
        teacher_attention_mask = torch.ones_like(teacher_input_ids)
        if pad_token_id is not None:
            teacher_attention_mask[teacher_input_ids == pad_token_id] = 0

        direction_signs = torch.tensor(direction_signs_list, device=device, dtype=torch.float32)

        # Compute group-relative GRPO advantages from all generated rollouts.
        if self.kl_mode in (-3, -4):
            advantages_list = []
            for p in range(P):
                n_pos = float(consensus_counts[p]) if consensus_counts[p] >= consensus_threshold else 0.0
                group_mean = n_pos / K_gen
                group_std = (n_pos * (1 - group_mean) ** 2 + (K_gen - n_pos) * group_mean ** 2) / K_gen
                group_std = group_std ** 0.5
                for j in range(K_train):
                    reward = 1.0 if direction_signs_list[p * K_train + j] > 0 else 0.0
                    adv = (reward - group_mean) / (group_std + 1e-6) if group_std > 1e-8 else (reward - group_mean)
                    advantages_list.append(adv)
            advantages = torch.tensor(advantages_list, device=device, dtype=torch.float32)
        else:
            advantages = None

        update_positive_count = int((direction_signs > 0).sum().item())
        update_negative_count = int((direction_signs < 0).sum().item())
        update_total = update_positive_count + update_negative_count
        consensus_mean = sum(consensus_counts) / len(consensus_counts) if consensus_counts else 0.0
        prompts_without_consensus = P - prompts_with_consensus
        extractable_fraction = pre_continuation_extractable / max(1, K_gen * P)
        backfill_fraction = backfill_count / max(1, K_train * P)

        overall_pos_count = 0
        for p in range(P):
            base = p * K_gen
            pseudo_label = pseudo_labels[p]
            if not pseudo_label or consensus_counts[p] < consensus_threshold:
                continue
            for k in range(K_gen):
                ans = all_answers[base + k]
                if ans and answers_equivalent(ans, pseudo_label):
                    overall_pos_count += 1

        vote_correct_count = 0
        vote_total = 0
        rollout_correct = 0
        rollout_total = 0
        pass_at_k_correct = 0
        pass_at_k_total = 0
        wrong_label_neither_count = 0
        wrong_label_rollout_total = 0
        for p in range(P):
            gt = gt_answers[p] if p < len(gt_answers) else None
            pl = pseudo_labels[p]
            if gt is None or gt == "":
                continue
            if pl:
                vote_total += 1
                if answers_equivalent(pl, gt):
                    vote_correct_count += 1
            base = p * K_gen
            wrong_label = bool(pl) and not answers_equivalent(pl, gt)
            any_correct = False
            for k in range(K_gen):
                ans = all_answers[base + k]
                if wrong_label:
                    wrong_label_rollout_total += 1
                    if not ans or (
                        not answers_equivalent(ans, pl) and not answers_equivalent(ans, gt)
                    ):
                        wrong_label_neither_count += 1
                if not ans:
                    continue
                rollout_total += 1
                if answers_equivalent(ans, gt):
                    rollout_correct += 1
                    any_correct = True
            pass_at_k_total += 1
            if any_correct:
                pass_at_k_correct += 1
        # Compare update directions against ground truth when labels are available.
        # A direction is correct iff (sign=+1 and answer ≡ GT) or (sign=-1 and answer ≢ GT).
        signal_correct = 0
        signal_total = 0
        update_signal_correct = 0
        update_signal_total = 0
        for p in range(P):
            gt = gt_answers[p] if p < len(gt_answers) else None
            if gt is None or gt == "":
                continue
            base = p * K_gen
            pseudo_label = pseudo_labels[p]
            has_consensus = bool(pseudo_label) and consensus_counts[p] >= consensus_threshold

            # All K_gen rollouts: inferred direction (vote rule, or the heuristic used
            # for no-consensus prompts: shortest extractable rollout is positive).
            if has_consensus:
                inferred_signs = [
                    1 if (all_answers[base + k] and answers_equivalent(all_answers[base + k], pseudo_label)) else -1
                    for k in range(K_gen)
                ]
            else:
                ext_ks = [k for k in range(K_gen) if all_answers[base + k] not in (None, "")]
                best_k = min(ext_ks, key=lambda k: comp_len(base + k)) if ext_ks else None
                inferred_signs = [1 if k == best_k else -1 for k in range(K_gen)]

            for k in range(K_gen):
                ans = all_answers[base + k]
                gt_sign = 1 if (ans and answers_equivalent(ans, gt)) else -1
                signal_total += 1
                if inferred_signs[k] == gt_sign:
                    signal_correct += 1

            # Selected K_train rollouts: actual training direction_signs (prompt-major layout).
            for j in range(K_train):
                pos = p * K_train + j
                gidx = selected_global_indices[pos]
                ans = all_answers[gidx]
                gt_sign = 1 if (ans and answers_equivalent(ans, gt)) else -1
                update_signal_total += 1
                if direction_signs_list[pos] == gt_sign:
                    update_signal_correct += 1

        sel_comp_lens = [len(completion_token_ids[i]) for i in selected_global_indices]

        # Aggregate numerators and denominators globally; do not average per-rank ratios.
        (
            g_vote_correct, g_vote_total,
            g_rollout_correct, g_rollout_total,
            g_pass_correct, g_pass_total,
            g_overall_pos, g_pos_denom,
            g_update_pos, g_update_total,
            g_pre_extractable, g_extract_denom,
            g_backfill, g_backfill_denom,
            g_consensus_sum, g_prompt_count,
            g_sel_len_sum, g_sel_len_count,
            g_prompts_with_consensus,
            g_update_neg,
            g_signal_correct, g_signal_total,
            g_update_signal_correct, g_update_signal_total,
            g_wrong_label_neither, g_wrong_label_total,
        ) = self._dist_reduce(
            [
                float(vote_correct_count), float(vote_total),
                float(rollout_correct), float(rollout_total),
                float(pass_at_k_correct), float(pass_at_k_total),
                float(overall_pos_count), float(K_gen * P),
                float(update_positive_count), float(update_total),
                float(pre_continuation_extractable), float(K_gen * P),
                float(backfill_count), float(K_train * P),
                float(sum(consensus_counts)), float(len(consensus_counts)),
                float(sum(sel_comp_lens)), float(len(sel_comp_lens)),
                float(prompts_with_consensus),
                float(update_negative_count),
                float(signal_correct), float(signal_total),
                float(update_signal_correct), float(update_signal_total),
                float(wrong_label_neither_count), float(wrong_label_rollout_total),
            ],
            op=ReduceOp.SUM,
        )
        g_sel_len_max = self._dist_reduce(
            [float(max(sel_comp_lens) if sel_comp_lens else 0)], op=ReduceOp.MAX
        )[0]

        vote_accuracy = g_vote_correct / g_vote_total if g_vote_total > 0 else 0.0
        rollout_accuracy = g_rollout_correct / g_rollout_total if g_rollout_total > 0 else 0.0
        pass_at_k = g_pass_correct / g_pass_total if g_pass_total > 0 else 0.0
        consensus_mean = g_consensus_sum / g_prompt_count if g_prompt_count > 0 else 0.0
        positive_fraction = g_overall_pos / g_pos_denom if g_pos_denom > 0 else 0.0
        update_positive_fraction = g_update_pos / g_update_total if g_update_total > 0 else 0.0
        extractable_fraction = g_pre_extractable / g_extract_denom if g_extract_denom > 0 else 0.0
        backfill_fraction = g_backfill / g_backfill_denom if g_backfill_denom > 0 else 0.0
        selected_completion_len_mean = g_sel_len_sum / g_sel_len_count if g_sel_len_count > 0 else 0.0
        signal_accuracy = g_signal_correct / g_signal_total if g_signal_total > 0 else 0.0
        update_signal_accuracy = g_update_signal_correct / g_update_signal_total if g_update_signal_total > 0 else 0.0
        wrong_label_neither_fraction = (
            g_wrong_label_neither / g_wrong_label_total if g_wrong_label_total > 0 else 0.0
        )

        self._metrics["train"]["ttrl/vote_accuracy"].append(vote_accuracy)
        self._metrics["train"]["ttrl/rollout_accuracy"].append(rollout_accuracy)
        self._metrics["train"]["ttrl/pass_at_k"].append(pass_at_k)
        self._metrics["train"]["ttrl/consensus_mean"].append(consensus_mean)
        self._metrics["train"]["ttrl/positive_fraction"].append(positive_fraction)
        self._metrics["train"]["ttrl/update_positive_fraction"].append(update_positive_fraction)
        self._metrics["train"]["ttrl/signal_accuracy"].append(signal_accuracy)
        self._metrics["train"]["ttrl/update_signal_accuracy"].append(update_signal_accuracy)
        self._metrics["train"]["ttrl/wrong_label_neither_fraction"].append(wrong_label_neither_fraction)
        self._metrics["train"]["ttrl/prompts_with_consensus"].append(g_prompts_with_consensus)
        self._metrics["train"]["oversample/extractable_fraction"].append(extractable_fraction)
        self._metrics["train"]["oversample/backfill_fraction"].append(backfill_fraction)
        self._metrics["train"]["oversample/selected_completion_len_mean"].append(selected_completion_len_mean)
        self._metrics["train"]["oversample/selected_completion_len_max"].append(g_sel_len_max)

        g_P = int(g_prompt_count)
        print(
            f"\n[TTPO Step {self.state.global_step}] (global) "
            f"P={g_P}, K_gen={K_gen}, K_train={K_train}, "
            f"extractable={extractable_fraction:.2%}, backfill={backfill_fraction:.2%}, "
            f"update_pos={int(g_update_pos)}, update_neg={int(g_update_neg)}, "
            f"overall_pos_frac={positive_fraction:.2%}, "
            f"update_pos_frac={update_positive_fraction:.2%}, "
            f"vote_acc={vote_accuracy:.2%}, rollout_acc={rollout_accuracy:.2%}, "
            f"signal_acc={signal_accuracy:.2%}, update_signal_acc={update_signal_accuracy:.2%}, "
            f"wrong_label_neither={wrong_label_neither_fraction:.2%}, "
            f"pass@{K_gen}={pass_at_k:.2%}, consensus_mean={consensus_mean:.1f}, "
            f"with_consensus={int(g_prompts_with_consensus)}/{g_P}"
        )

        # Reorder from prompt-major to rollout-major.
        reorder_idx = torch.arange(K_train * P, device=device).view(P, K_train).t().reshape(-1)
        student_input_ids = student_input_ids[reorder_idx]
        student_attention_mask = student_attention_mask[reorder_idx]
        teacher_input_ids = teacher_input_ids[reorder_idx]
        teacher_attention_mask = teacher_attention_mask[reorder_idx]
        labels = labels[reorder_idx]
        direction_signs = direction_signs[reorder_idx]
        if advantages is not None:
            advantages = advantages[reorder_idx]

        all_items = {
            "student_input_ids": student_input_ids,
            "student_attention_mask": student_attention_mask,
            "student_prompt_length": max_student_prompt_len,
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_prompt_length": max_teacher_prompt_len,
            "labels": labels,
            "direction_signs": direction_signs,
        }
        if advantages is not None:
            all_items["advantages"] = advantages

        gidx_to_selected_pos = {gidx: pos for pos, gidx in enumerate(selected_global_indices)}

        log_prompts: list[str] = []
        log_completions: list[str] = []
        log_rollout_idx: list[int] = []
        log_answers: list[str] = []
        log_is_selected: list[bool] = []
        log_is_majority: list[bool] = []
        log_direction_sign: list[int] = []
        log_ground_truth: list[str] = []

        for p in range(P):
            base = p * K_gen
            pseudo_label = pseudo_labels[p]
            gt_raw = gt_answers[p] if p < len(gt_answers) else None
            gt_str = gt_raw if gt_raw else ""
            for k in range(K_gen):
                gidx = base + k
                ans = all_answers[gidx]
                ans_str = ans if ans is not None else ""
                sel_pos = gidx_to_selected_pos.get(gidx)
                is_selected = sel_pos is not None
                direction_sign = int(direction_signs_list[sel_pos]) if is_selected else 0
                is_majority = bool(
                    pseudo_label and ans_str and answers_equivalent(ans_str, pseudo_label)
                )
                log_prompts.append(prompts_text[p])
                log_completions.append(completion_texts[gidx])
                log_rollout_idx.append(k)
                log_answers.append(ans_str)
                log_is_selected.append(is_selected)
                log_is_majority.append(is_majority)
                log_direction_sign.append(direction_sign)
                log_ground_truth.append(gt_str)

        self._textual_logs["prompt"].extend(gather_object(log_prompts))
        self._textual_logs["completion"].extend(gather_object(log_completions))
        self._textual_logs["rollout_idx"].extend(gather_object(log_rollout_idx))
        self._textual_logs["answer"].extend(gather_object(log_answers))
        self._textual_logs["is_selected"].extend(gather_object(log_is_selected))
        self._textual_logs["is_majority"].extend(gather_object(log_is_majority))
        self._textual_logs["direction_sign"].extend(gather_object(log_direction_sign))
        self._textual_logs["ground_truth"].extend(gather_object(log_ground_truth))

        for prompt_str, completion_str, r_idx, ans_str, sel, maj, ds, gt_str in zip(
            log_prompts,
            log_completions,
            log_rollout_idx,
            log_answers,
            log_is_selected,
            log_is_majority,
            log_direction_sign,
            log_ground_truth,
        ):
            self._generation_outputs_buffer.append(
                {
                    "step": self.state.global_step,
                    "prompt": prompt_str,
                    "completion": completion_str,
                    "rollout_idx": r_idx,
                    "answer": ans_str,
                    "ground_truth": gt_str,
                    "is_selected": sel,
                    "is_majority": maj,
                    "direction_sign": ds,
                }
            )

        if random.random() < 0.01 and len(log_prompts) > 0:
            sample_idx = random.randint(0, len(log_prompts) - 1)
            print(f"\n{'='*80}")
            print(f"TTPO (oversample-and-select) GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"Prompt:\n{log_prompts[sample_idx]}")
            print(
                f"\nRollout {log_rollout_idx[sample_idx]} | "
                f"answer={log_answers[sample_idx]!r} | "
                f"selected={log_is_selected[sample_idx]} | "
                f"majority={log_is_majority[sample_idx]} | "
                f"direction_sign={log_direction_sign[sample_idx]}"
            )
            print(f"\nCompletion:\n{log_completions[sample_idx]}")
            print(f"{'='*80}\n")

        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        return all_items

    def _training_step_single(
        self, inputs: dict[str, torch.Tensor | Any], model: nn.Module, num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        raise NotImplementedError(
            "The all-rollout contrastive TTPO design requires num_rollouts > 1 (majority vote "
            "is needed to assign positive/negative directions). Set num_rollouts >= 2."
        )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                    "prompt": list(self._textual_logs["prompt"]),
                    "completion": list(self._textual_logs["completion"]),
                    "rollout_idx": list(self._textual_logs["rollout_idx"]),
                    "answer": list(self._textual_logs["answer"]),
                    "ground_truth": list(self._textual_logs["ground_truth"]),
                    "is_selected": list(self._textual_logs["is_selected"]),
                    "is_majority": list(self._textual_logs["is_majority"]),
                    "direction_sign": list(self._textual_logs["direction_sign"]),
                }
                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt", "rollout_idx"])
                if self.num_completions_to_print and len(df) > 0:
                    df = df.sample(n=min(self.num_completions_to_print, len(df)), random_state=42)
                wandb.log({"completions": wandb.Table(dataframe=df)}, commit=False)
