from __future__ import annotations

import time
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

from ..models import Example, Prediction


def laya_request_tokens(
    question: dict[str, Any],
    text: str,
    tokenizer: Any,
    to_internal: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    render_options: Callable[[dict[str, Any]], list[str]] | None = None,
    serialize_state: Callable[[dict[str, str]], str] | None = None,
) -> tuple[list[int], int, int, int]:
    """Count exactly the tokens consumed by Laya's option and state renderers."""
    if render_options is None or serialize_state is None:
        common = import_module("laya.common")
        render_options = common.render_options
        serialize_state = common.serialize_state
    assert render_options is not None and serialize_state is not None
    internal = to_internal(question)
    mask = tokenizer.mask_token
    options = render_options(internal)
    option_sizes = [
        1 + len(tokenizer(" " + option.replace(mask, " "), add_special_tokens=False)["input_ids"])
        for option in options
    ]
    instruction_tokens = len(
        tokenizer(
            f"{internal['t']} question: {str(internal['ins']).replace(mask, ' ')}",
            add_special_tokens=False,
        )["input_ids"]
    )
    text_tokens = len(
        tokenizer(serialize_state({"text": text}).replace(mask, " "), add_special_tokens=False)[
            "input_ids"
        ]
    )
    return (
        option_sizes,
        instruction_tokens,
        text_tokens,
        3 + sum(option_sizes) + instruction_tokens + text_tokens,
    )


def laya_head_fits(option_sizes: list[int], instruction_tokens: int, head: int) -> bool:
    """Mirror laya.common.build_sequence: each option is cut at 48 tokens plus its mask, every
    option is trimmed once ``head - sum(options) < 16``, and the instruction keeps the rest."""
    return (
        all(size <= 49 for size in option_sizes)
        and sum(option_sizes) + max(instruction_tokens, 16) <= head
    )


class LayaBackend:
    name = "laya"

    def __init__(
        self,
        model_id: str,
        revision: str,
        local_repo: str,
        checkpoint: str,
        head_max_len: dict[str, int],
        questions: dict[str, str],
        *,
        agent: Any = None,
        synchronize: Callable[[], None] | None = None,
        latency_paraphrases: list[str] | None = None,
        backend_name: str = "laya",
    ) -> None:
        self.name = backend_name
        self.model_id = model_id
        self.revision = revision
        self.checkpoint = checkpoint
        self.head_max_len = head_max_len
        self.questions = questions
        self.latency_paraphrases = latency_paraphrases or []
        self.model_latency_seconds = 0.0
        if agent is None:
            import torch

            runtime = import_module("laya")
            agent_type = runtime.Agent
            router_type = runtime.Router

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            if checkpoint == "multilingual":
                router = router_type(
                    models={"multilingual": str(Path(local_repo) / "multilingual")}, device=device
                )
                agent = router.load("multilingual")
            else:
                agent = agent_type(
                    local_repo,
                    device=device,
                    subfolder="typed-decisions" if checkpoint == "typed-decisions" else None,
                )
            if synchronize is None:
                synchronize = torch.mps.synchronize if device == "mps" else lambda: None
        self.agent = agent
        self.synchronize = synchronize or (lambda: None)
        self.device = str(agent.device)
        self.default_head_max_len = int(agent.cfg.get("head_max_len", 192))

    def warmup(self, example: Example) -> None:
        self.predict("warmup", example)

    def _question(self, example: Example) -> dict[str, Any]:
        return self.question_for(example, self.questions)

    @staticmethod
    def question_for(example: Example, questions: dict[str, str]) -> dict[str, Any]:
        order = example.option_order or tuple(range(len(example.labels)))
        if example.question_type == "choice":
            criteria: object = {f"label_{index:03d}": example.labels[index] for index in order}
        elif example.question_type == "score":
            criteria = [example.labels[index] for index in order]
        else:
            criteria = {"false": example.labels[0], "true": example.labels[1]}
        return {
            "type": example.question_type,
            "instructions": example.instructions or questions[example.question_type],
            "criteria": criteria,
        }

    def _assert_untruncated(self, example: Example, question: dict[str, Any]) -> None:
        option_sizes, instructions, _, total = laya_request_tokens(
            question, example.text, self.agent.tok, self.agent._to_internal
        )
        if any(size > 49 for size in option_sizes):
            raise ValueError("Laya option would exceed its 48-token per-option cap")
        head = self.head_max_len[example.dataset]
        if head < self.default_head_max_len:
            raise ValueError("Laya primary head is below the shipped head_max_len")
        if not laya_head_fits(option_sizes, instructions, head):
            raise ValueError("Laya option head would truncate")
        if total > self.agent.cfg.get("max_len", 512):
            raise ValueError("Laya full request would truncate text")

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        question = self._question(example)
        default_head = example.permutation_id == "head-default"
        if not default_head:
            self._assert_untruncated(example, question)
        questions = {"label": question}
        if example.permutation_id == "latency-10":
            if len(self.latency_paraphrases) != 10:
                raise ValueError("latency-10 requires ten frozen paraphrases")
            questions = {
                f"label_{index}": {**question, "instructions": paraphrase}
                for index, paraphrase in enumerate(self.latency_paraphrases)
            }
            for item in questions.values():
                self._assert_untruncated(example, item)
        head = self.default_head_max_len if default_head else self.head_max_len[example.dataset]
        self.agent.cfg["head_max_len"] = head
        timed_target = self.agent.model if hasattr(self.agent, "model") else self.agent
        method_name = "forward" if timed_target is not self.agent else "_forward"
        original_forward = getattr(timed_target, method_name)

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            self.synchronize()
            start = time.perf_counter()
            result = original_forward(*args, **kwargs)
            self.synchronize()
            self.model_latency_seconds = time.perf_counter() - start
            return result

        setattr(timed_target, method_name, timed_forward)
        start = time.perf_counter()
        try:
            result = self.agent.system_one({"text": example.text}, questions)
        finally:
            setattr(timed_target, method_name, original_forward)
        answer = result["answers"]["label_0" if example.permutation_id == "latency-10" else "label"]
        if example.question_type == "choice":
            probabilities = tuple(
                float(answer["probabilities"][f"label_{index:03d}"])
                for index in range(len(example.labels))
            )
        elif example.question_type == "noul":
            positive = float(answer["noul"])
            probabilities = (1.0 - positive, positive)
        else:
            values = answer.get("probabilities")
            if values:
                order = example.option_order or tuple(range(len(example.labels)))
                mapped = [0.0] * len(order)
                for position, index in enumerate(order):
                    mapped[index] = float(values[str(position)])
                probabilities = tuple(mapped)
            else:
                probabilities = ()
        expected = (
            sum((index + 1) * value for index, value in enumerate(probabilities))
            if probabilities and example.question_type == "score"
            else None
        )
        if expected is None and example.question_type == "score":
            expected = float(answer["score"]) + 1
        predicted = (
            max(range(len(probabilities)), key=probabilities.__getitem__)
            if probabilities
            else int(expected + 0.5) - 1
            if expected is not None
            else -1
        )
        return Prediction(
            experiment_id,
            self.name,
            self.model_id,
            f"{self.model_id}@{self.revision}:{self.checkpoint}",
            example.dataset,
            example.example_id,
            example.target_index,
            predicted,
            example.labels,
            probabilities,
            time.perf_counter() - start,
            question_type=example.question_type,
            split=example.split,
            permutation_id=example.permutation_id,
            letter_mode=example.letter_mode,
            raw_probabilities=probabilities,
            expected_score=expected,
            model_latency_seconds=self.model_latency_seconds,
            device=str(self.agent.device),
            resolved_checkpoint=self.checkpoint,
            head_max_len_used=head,
        )

    def close(self) -> None:
        return None
