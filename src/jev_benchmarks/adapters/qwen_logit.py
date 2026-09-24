from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import Any, cast

from ..models import Example, Prediction


def softmax_scores(scores: Sequence[float]) -> tuple[float, ...]:
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def two_digit_joint(
    first_log_probs: dict[int, float],
    second_log_probs: dict[int, dict[int, float]],
    suffixes: Sequence[tuple[int, int]],
) -> tuple[float, ...]:
    return softmax_scores(
        [first_log_probs[first] + second_log_probs[first][second] for first, second in suffixes]
    )


def qwen_option_ids(count: int) -> tuple[str, list[str]]:
    """Return the assistant prefill and the scored option-ID suffixes.

    Qwen merges a space with a following letter (" A" is one token) but splits digits, so
    letters carry their own space after "Answer:" and two-digit IDs follow "Answer: ".
    """
    if count <= 26:
        return "Answer:", [f" {chr(65 + index)}" for index in range(count)]
    return "Answer: ", [f"{index + 1:02d}" for index in range(count)]


class _HFScorer:
    def __init__(self, model_id: str, revision: str, local_path: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, revision=revision, local_files_only=True, trust_remote_code=False
        )
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        self.model = (
            cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    local_path,
                    revision=revision,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                    torch_dtype=dtype,
                ),
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.model_seconds = 0.0

    def _logits(self, token_batches: list[list[int]]) -> Any:
        torch = self.torch
        tensor = torch.tensor(token_batches, device=self.device)
        if self.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_ids=tensor).logits.float()
        if self.device == "mps":
            torch.mps.synchronize()
        self.model_seconds = getattr(self, "model_seconds", 0.0) + time.perf_counter() - start
        return logits

    def hidden(self, prompt: str, layer: int) -> list[float]:
        """Output of decoder block ``layer`` at the last prompt position (fp32)."""
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        with self.torch.no_grad():
            output = self.model(
                input_ids=self.torch.tensor([ids], device=self.device), output_hidden_states=True
            )
        # hidden_states[0] is the embedding output, so index ``layer`` is block ``layer``.
        return output.hidden_states[layer][0, -1].float().cpu().tolist()

    def __call__(self, prompt: str, ids: Sequence[str], mode: str) -> tuple[float, ...]:
        tokenizer = self.tokenizer
        prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        suffixes: list[list[int]] = []
        complete_ids: list[list[int]] = []
        for label_id in ids:
            complete = tokenizer(prompt + label_id, add_special_tokens=False)["input_ids"]
            complete_ids.append(complete)
            if mode in {"letter", "two_digit_joint"} and complete[: len(prefix)] != prefix:
                raise ValueError("Qwen option ID is not a suffix of the frozen rendered prompt")
            suffixes.append(complete[len(prefix) :])
        if mode == "letter" and any(len(suffix) != 1 for suffix in suffixes):
            raise ValueError("Qwen letter mode requires one suffix token per ID")
        if mode == "two_digit_joint" and any(len(suffix) != 2 for suffix in suffixes):
            raise ValueError("Qwen two-digit mode requires two suffix tokens per ID")
        torch = self.torch
        if mode == "full_sequence":
            scores = []
            for complete in complete_ids:
                if len(complete) < 2:
                    raise ValueError("complete Qwen prompt must contain at least two tokens")
                logits = self._logits([complete[:-1]])[0]
                log_probs = torch.log_softmax(logits, dim=-1)
                scores.append(
                    sum(
                        float(log_probs[index - 1, token])
                        for index, token in enumerate(complete)
                        if index
                    )
                )
            return tuple(scores)
        if mode == "letter":
            first = torch.log_softmax(self._logits([prefix])[0, -1], dim=-1)
            return tuple(float(first[suffix[0]]) for suffix in suffixes)
        if mode == "two_digit_joint":
            digit1 = sorted({suffix[0] for suffix in suffixes})
            if self.device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                initial = self.model(
                    input_ids=torch.tensor([prefix], device=self.device), use_cache=True
                )
            if self.device == "mps":
                torch.mps.synchronize()
            self.model_seconds = getattr(self, "model_seconds", 0.0) + time.perf_counter() - start
            first = torch.log_softmax(initial.logits[0, -1].float(), dim=-1)
            cache = initial.past_key_values
            if cache is None or not hasattr(cache, "batch_repeat_interleave"):
                raise ValueError("Qwen model did not return a reusable KV cache")
            cache.batch_repeat_interleave(len(digit1))
            if self.device == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                continuation = self.model(
                    input_ids=torch.tensor([[digit] for digit in digit1], device=self.device),
                    attention_mask=torch.ones((len(digit1), len(prefix) + 1), device=self.device),
                    past_key_values=cache,
                    use_cache=True,
                )
            if self.device == "mps":
                torch.mps.synchronize()
            self.model_seconds += time.perf_counter() - start
            next_probs = torch.log_softmax(continuation.logits[:, -1].float(), dim=-1)
            lookup = {digit: index for index, digit in enumerate(digit1)}
            return tuple(
                float(first[suffix[0]] + next_probs[lookup[suffix[0]], suffix[1]])
                for suffix in suffixes
            )
        raise ValueError(f"unknown Qwen score mode: {mode}")

    def batch(self, prompt: str, ids: Sequence[str], mode: str, count: int) -> tuple[float, ...]:
        if mode == "full_sequence":
            scores = []
            for label_id in ids:
                complete = self.tokenizer(prompt + label_id, add_special_tokens=False)["input_ids"]
                logits = self._logits([complete[:-1]] * count)[0]
                log_probs = self.torch.log_softmax(logits, dim=-1)
                scores.append(
                    sum(
                        float(log_probs[index - 1, token])
                        for index, token in enumerate(complete)
                        if index
                    )
                )
            return tuple(scores)
        if mode != "letter":
            raise ValueError("latency batch supports AG News letter or full-sequence scoring")
        prefix = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        suffixes = []
        for label_id in ids:
            full = self.tokenizer(prompt + label_id, add_special_tokens=False)["input_ids"]
            suffix = full[len(prefix) :]
            if full[: len(prefix)] != prefix or len(suffix) != 1:
                raise ValueError("latency letter suffix is not one token")
            suffixes.append(suffix[0])
        logits = self._logits([prefix] * count)
        log_probs = self.torch.log_softmax(logits[0, -1], dim=-1)
        return tuple(float(log_probs[token]) for token in suffixes)


class QwenLogitBackend:
    name = "qwen_logit"

    def __init__(
        self,
        model_id: str,
        revision: str,
        local_path: str,
        system_prompt: str,
        user_template: str,
        score_modes: dict[str, str],
        *,
        scorer: Callable[[str, Sequence[str], str], Sequence[float]] | None = None,
        renderer: Callable[[str, str], str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.score_modes = score_modes
        self.scorer = scorer or _HFScorer(model_id, revision, local_path)
        if renderer is None:
            tokenizer = self.scorer.tokenizer  # type: ignore[attr-defined]

            def default_renderer(system: str, user: str) -> str:
                return tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

            renderer = default_renderer

        self.renderer = renderer
        self.device = getattr(self.scorer, "device", "fake")

    def warmup(self, example: Example) -> None:
        self.predict("warmup", example)

    def prompt(self, example: Example) -> tuple[str, list[str]]:
        """Rendered prompt ending in the answer prefill, and the scored option-ID suffixes."""
        order = example.option_order or tuple(range(len(example.labels)))
        prefill, labels = qwen_option_ids(len(example.labels))
        lines = []
        for position, index in enumerate(order):
            label_id = labels[index] if example.letter_mode == "stable" else labels[position]
            lines.append(f"{label_id.strip()}) {example.labels[index]}")
        user = self.user_template.format(
            text=example.text, options="\n".join(lines), question=example.instructions
        )
        return self.renderer(self.system_prompt, user) + prefill, labels

    def features(self, example: Example, layer: int) -> list[float]:
        return self.scorer.hidden(self.prompt(example)[0], layer)  # type: ignore[attr-defined]

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        order = example.option_order or tuple(range(len(example.labels)))
        prompt, labels = self.prompt(example)
        mode = self.score_modes[example.dataset]
        if mode not in {"letter", "two_digit_joint", "full_sequence"}:
            raise ValueError("Qwen score mode needs TO-FILL-AFTER-PROBE resolution")
        if hasattr(self.scorer, "model_seconds"):
            self.scorer.model_seconds = 0.0  # type: ignore[attr-defined]
        start = time.perf_counter()
        if example.permutation_id == "latency-10" and hasattr(self.scorer, "batch"):
            log_scores = self.scorer.batch(prompt, labels, mode, 10)  # type: ignore[attr-defined]
        elif example.permutation_id == "latency-10":
            raise ValueError("latency-10 requires a batched scorer")
        else:
            log_scores = self.scorer(prompt, labels, mode)
        elapsed = time.perf_counter() - start
        scored = softmax_scores(log_scores)
        if example.letter_mode == "positional":
            mapped = [0.0] * len(scored)
            for position, index in enumerate(order):
                mapped[index] = scored[position]
            probabilities = tuple(mapped)
        else:
            probabilities = scored
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        expected = (
            sum((index + 1) * value for index, value in enumerate(probabilities))
            if example.question_type == "score"
            else None
        )
        return Prediction(
            experiment_id,
            self.name,
            self.model_id,
            f"{self.model_id}@{self.revision}",
            example.dataset,
            example.example_id,
            example.target_index,
            predicted,
            example.labels,
            probabilities,
            elapsed,
            question_type=example.question_type,
            split=example.split,
            permutation_id=example.permutation_id,
            letter_mode=example.letter_mode,
            raw_probabilities=probabilities,
            expected_score=expected,
            model_latency_seconds=getattr(self.scorer, "model_seconds", elapsed),
            device=self.device,
            resolved_checkpoint=self.revision,
        )

    def close(self) -> None:
        return None
