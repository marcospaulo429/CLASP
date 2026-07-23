"""Answer generators for the SVQ RAG eval.

``HFGenerator`` runs a local instruct LLM (default Qwen3-8B) via transformers.
``DryRunGenerator`` is a weightless stub for testing the pipeline end-to-end
without loading an LLM (used by ``--dry-run-generator``).
"""

from __future__ import annotations

from typing import Protocol, Sequence

_SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the question using ONLY the "
    "information in the provided passages. Reply with the exact answer, as short "
    "as possible (a word or short phrase). If the passages do not contain the "
    "answer, reply exactly: No Answer."
)


def build_prompt(question: str, passages: Sequence[str]) -> str:
    context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"Passages:\n{context}\n\nQuestion: {question}\nAnswer:"


class Generator(Protocol):
    def generate(self, question: str, passages: Sequence[str], language: str | None = None) -> str:
        ...


class DryRunGenerator:
    """No-LLM stub: echoes the start of the top passage. Deterministic, no weights."""

    name = "dry-run"

    def generate(self, question: str, passages: Sequence[str], language: str | None = None) -> str:
        if not passages:
            return "No Answer"
        return " ".join(passages[0].split()[:8])


class HFGenerator:
    """Local HuggingFace causal-LM generator (default Qwen3-8B)."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        device: str | None = None,
        max_new_tokens: int = 64,
        enable_thinking: bool = False,
        dtype: str = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        torch_dtype = getattr(torch, dtype) if dtype not in ("auto", None) else "auto"
        model_kwargs: dict = {"torch_dtype": torch_dtype}
        if device:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(device)
        else:
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.eval()

    def _apply_chat_template(self, question: str, passages: Sequence[str]) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, passages)},
        ]
        try:
            # Qwen3 supports `enable_thinking`; ignored by templates that don't.
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def generate(self, question: str, passages: Sequence[str], language: str | None = None) -> str:
        import torch

        text = self._apply_chat_template(question, passages)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs.input_ids.shape[1]:]
        out = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        # Strip any residual Qwen3 <think>...</think> block if thinking leaked in.
        if "</think>" in out:
            out = out.split("</think>", 1)[1].strip()
        return out.splitlines()[0].strip() if out else "No Answer"
