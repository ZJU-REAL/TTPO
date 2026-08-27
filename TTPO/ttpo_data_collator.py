import torch


class TTPODataCollator:
    """
    Data collator for TTPO (TTRL + Self-Distillation).

    Only creates student prompts — no teacher prompts or solution fields needed,
    since the teacher distribution is constructed from voting on the student's own logits.
    """

    def __init__(self, tokenizer, max_length=2048, student_thinking=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.student_thinking = student_thinking
        self.tokenizer.padding_side = "right"

    def __call__(self, features):
        student_prompts = []
        problems = []
        answers = []

        for feature in features:
            problem = feature["problem"]
            problems.append(problem)
            answer = feature.get("Answer", feature.get("answer"))
            answers.append(None if answer is None else str(answer))
            user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            messages = [{"role": "user", "content": user_message}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.student_thinking
            )
            student_prompts.append(prompt)

        encoded_no_pad = self.tokenizer(
            student_prompts, padding=False, truncation=True, max_length=self.max_length
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths)

        encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )

        return {
            "student_prompts": encoded["input_ids"],
            "student_prompt_attention_mask": encoded["attention_mask"],
            "student_prompt_length": max_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(prompt_lengths),
            "problems": problems,
            "answers": answers,
        }
