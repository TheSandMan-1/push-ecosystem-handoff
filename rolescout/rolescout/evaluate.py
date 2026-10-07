"""Measure extraction accuracy against hand-labeled postings.

    rolescout eval                         # rules only
    rolescout eval --extractor local       # your Ollama / LM Studio model
    rolescout eval --extractor anthropic --model claude-sonnet-5-5

Use this to decide which model is good enough for the bulk tier, and to prove a
prompt or rule change helped before shipping it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .config import Settings
from .extract import heuristics
from .extract.heuristics import Facts
from .extract.pipeline import FACT_FIELDS, _call_validated, apply_guardrails, facts_from_model
from .extract.prompts import EXTRACT_SYSTEM, job_block
from .extract.providers import LLMError, Provider
from .extract.schema import ExtractedFacts, facts_json_schema


def load_cases(path: str | None = None) -> list[dict[str, Any]]:
    if path:
        text = Path(path).read_text()
    else:
        text = resources.files("rolescout").joinpath("seeds/eval_jobs.yaml").read_text()
    return yaml.safe_load(text) or []


@dataclass
class EvalReport:
    label: str
    cases: int = 0
    checks: int = 0
    correct: int = 0
    per_field: dict[str, list[int]] = field(default_factory=dict)  # field -> [correct, total]
    failures: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    latency_ms: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.checks if self.checks else 0.0


def extract_one(case: dict, provider: Provider | None, settings: Settings) -> tuple[Facts, float, int]:
    rules = heuristics.extract(case["title"], case.get("description"), case.get("location"))
    if provider is None:
        return rules, 0.0, 0
    text = job_block(case["title"], case.get("company", "Example Co"), case.get("location"),
                     case.get("description"), settings.llm_max_description_chars)
    result, parsed = _call_validated(provider, EXTRACT_SYSTEM, text, facts_json_schema(), "job_facts", ExtractedFacts)
    return apply_guardrails(rules, facts_from_model(parsed)), result.cost_usd, result.latency_ms


def run_eval(cases: list[dict], provider: Provider | None, settings: Settings) -> EvalReport:
    label = f"{provider.name}:{provider.model}" if provider else "rules"
    report = EvalReport(label=label)
    for case in cases:
        report.cases += 1
        try:
            facts, cost, latency = extract_one(case, provider, settings)
        except LLMError as exc:
            report.errors.append(f"{case['id']}: {exc}")
            continue
        report.cost_usd += cost
        report.latency_ms += latency
        for name, expected in (case.get("expect") or {}).items():
            if name not in FACT_FIELDS:
                continue
            got = getattr(facts, name)
            ok = got == expected
            bucket = report.per_field.setdefault(name, [0, 0])
            bucket[1] += 1
            report.checks += 1
            if ok:
                bucket[0] += 1
                report.correct += 1
            else:
                report.failures.append(f"{case['id']}: {name} expected {expected!r}, got {got!r}")
    return report


def format_report(report: EvalReport) -> str:
    lines = [f"Extractor: {report.label}",
             f"Cases: {report.cases}   Checks: {report.checks}   Accuracy: {report.accuracy:.1%}"]
    for name, (ok, total) in sorted(report.per_field.items()):
        lines.append(f"  {name:20} {ok}/{total}  ({ok / total:.0%})")
    if report.cost_usd or report.latency_ms:
        lines.append(f"Cost: ${report.cost_usd:.4f}   Avg latency: {report.latency_ms / max(report.cases, 1):.0f} ms")
    if report.failures:
        lines.append("Failures:")
        lines += [f"  - {f}" for f in report.failures]
    if report.errors:
        lines.append("Errors:")
        lines += [f"  - {e}" for e in report.errors]
    return "\n".join(lines)
