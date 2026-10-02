"""Label-blind teacher rationale prompting and JSONL caching."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 48
MAX_NEW_TOKENS_LIMIT = 128
RATIONALE_PROMPT_VERSION = "eedi-error-rationale-v1"


def _required_text(record: Mapping[str, Any], field: str, row_number: int = 1) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"record {row_number} has an empty {field}")
    return value.strip()


def build_rationale_prompt(record: Mapping[str, Any]) -> str:
    """Build a teacher prompt solely from the question and its answer texts."""

    if not isinstance(record, Mapping):
        raise ValueError("rationale input must be a mapping")
    question = _required_text(record, "question")
    correct_answer = _required_text(record, "correct_answer")
    distractor = _required_text(record, "distractor")
    return (
        "You are a math tutor. Infer the likely reasoning behind the incorrect answer using only the question, correct answer, and distractor. Do not name a misconception category. Return one concise sentence describing the reasoning error.\n\n"
        f"Question: {question}\n"
        f"Correct answer: {correct_answer}\n"
        f"Distractor: {distractor}\n\n"
        "Reasoning error:"
    )


def build_rationale_cache_record(
    record: Mapping[str, Any],
    rationale: str,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    teacher_revision: str | None = None,
    prompt_version: str = RATIONALE_PROMPT_VERSION,
) -> dict[str, str]:
    """Create a cache record with a stable SHA-256 over the full prompt."""

    if not isinstance(record, Mapping):
        raise ValueError("rationale input must be a mapping")
    query_id = record.get("query_id")
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("rationale input has an invalid query_id")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be non-empty text")
    if not isinstance(teacher_model, str) or not teacher_model.strip():
        raise ValueError("teacher_model must be non-empty text")
    if teacher_revision is None:
        teacher_revision = "unresolved-at-load"
    if not isinstance(teacher_revision, str) or not teacher_revision.strip():
        raise ValueError("teacher_revision must be non-empty text")
    if not isinstance(prompt_version, str) or not prompt_version.strip():
        raise ValueError("prompt_version must be non-empty text")
    prompt = build_rationale_prompt(record)
    input_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "query_id": query_id.strip(),
        "rationale": rationale.strip(),
        "teacher_model": teacher_model.strip(),
        "teacher_revision": teacher_revision.strip(),
        "prompt_version": prompt_version.strip(),
        "input_sha256": input_sha256,
    }


def _read_cache(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    cache: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"rationale cache line {line_number} is not valid JSON") from error
            required_fields = (
                "query_id",
                "rationale",
                "teacher_model",
                "teacher_revision",
                "prompt_version",
                "input_sha256",
            )
            if not isinstance(record, Mapping) or any(
                not isinstance(record.get(field), str) for field in required_fields
            ):
                raise ValueError(f"rationale cache line {line_number} is missing required fields")
            if any(not record[field].strip() for field in required_fields):
                raise ValueError(f"rationale cache line {line_number} has an empty required field")
            key = (
                record["query_id"],
                record["teacher_model"],
                record["teacher_revision"],
                record["prompt_version"],
                record["input_sha256"],
            )
            cache[key] = dict(record)
    return cache


def _cache_lookup(
    cache: Mapping[tuple[str, str, str, str, str], dict[str, str]],
    *,
    query_id: str,
    teacher_model: str,
    teacher_revision: str | None,
    input_sha256: str,
) -> dict[str, str] | None:
    for key, record in reversed(list(cache.items())):
        cached_query, cached_model, cached_revision, cached_prompt, cached_hash = key
        if (
            cached_query == query_id
            and cached_model == teacher_model
            and cached_prompt == RATIONALE_PROMPT_VERSION
            and cached_hash == input_sha256
            and (teacher_revision is None or cached_revision == teacher_revision)
        ):
            return record
    return None


def _model_revision(model: Any, requested_revision: str | None) -> str:
    resolved = getattr(getattr(model, "config", None), "_commit_hash", None)
    return str(resolved or requested_revision or "unresolved-at-load")


def _render_teacher_text(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _write_cache(path: Path, records: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def generate_rationales(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    model_name: str = DEFAULT_TEACHER_MODEL,
    revision: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    device: str | None = None,
    batch_size: int = 1,
    progress: bool = False,
) -> list[dict[str, str]]:
    """Generate deterministic, short rationales and persist versioned JSONL.

    Existing cache entries are reused when their query input, model, and prompt
    identity match. A complete cache can be read without loading the 7B model.
    """

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS_LIMIT}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be non-empty text")
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("revision must be non-empty text or None")

    rows = list(records)
    prompts: list[str] = []
    query_ids: list[str] = []
    input_hashes: list[str] = []
    seen_ids: set[str] = set()
    for row_number, record in enumerate(rows, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"rationale record {row_number} must be a mapping")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"rationale record {row_number} has an invalid query_id")
        if query_id in seen_ids:
            raise ValueError(f"rationale query_id {query_id!r} occurs more than once")
        seen_ids.add(query_id)
        prompt = build_rationale_prompt(record)
        prompts.append(prompt)
        query_ids.append(query_id)
        input_hashes.append(hashlib.sha256(prompt.encode("utf-8")).hexdigest())

    output_file = Path(output_path)
    existing_cache = _read_cache(output_file)
    if not rows:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("", encoding="utf-8")
        return []

    cached_rows = [
        _cache_lookup(
            existing_cache,
            query_id=query_id,
            teacher_model=model_name,
            teacher_revision=revision,
            input_sha256=input_sha256,
        )
        for query_id, input_sha256 in zip(query_ids, input_hashes)
    ]
    if all(cached_rows):
        results = [dict(record) for record in cached_rows if record is not None]
        _write_cache(output_file, results)
        return results

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("rationale generation requires torch and transformers") from error

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if target_device.type == "cuda" else torch.float32
    }
    if revision is not None:
        model_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_name, **({"revision": revision} if revision else {}))
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(target_device)
    model.eval()
    resolved_revision = _model_revision(model, revision)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("teacher tokenizer must define an EOS or padding token")

    # Left padding keeps every prompt's last token adjacent to its continuation.
    tokenizer.padding_side = "left"
    results: list[dict[str, str] | None] = [
        _cache_lookup(
            existing_cache,
            query_id=query_id,
            teacher_model=model_name,
            teacher_revision=resolved_revision,
            input_sha256=input_sha256,
        )
        for query_id, input_sha256 in zip(query_ids, input_hashes)
    ]
    pending = [index for index, cached in enumerate(results) if cached is None]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        texts = [_render_teacher_text(tokenizer, prompts[index]) for index in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = encoded["input_ids"].to(target_device)
        attention_mask = encoded["attention_mask"].to(target_device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, index in enumerate(batch):
            continuation = generated[row, input_ids.shape[-1] :]
            rationale = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            if not rationale:
                raise ValueError(f"teacher generated an empty rationale for {query_ids[index]!r}")
            results[index] = build_rationale_cache_record(
                rows[index],
                rationale,
                teacher_model=model_name,
                teacher_revision=resolved_revision,
            )
        # Checkpoint after each batch so an interrupted run resumes from the cache.
        _write_cache(output_file, [record for record in results if record is not None])
        if progress:
            print(f"rationales {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)

    final = [record for record in results if record is not None]
    _write_cache(output_file, final)
    return final
