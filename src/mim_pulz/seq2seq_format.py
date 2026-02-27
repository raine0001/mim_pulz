from __future__ import annotations


DEFAULT_TASK_PREFIX = "translate transliteration to english:"


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def normalize_task_prefix(task_prefix: str | None) -> str:
    if task_prefix is None:
        return ""
    prefix = " ".join(str(task_prefix).split()).strip()
    if not prefix:
        return ""
    if not prefix.endswith(" "):
        prefix += " "
    return prefix


def normalize_domain_tag(domain: str | None) -> str:
    if domain is None:
        return ""
    d = " ".join(str(domain).split()).strip().lower()
    if not d:
        return ""
    return f"<domain={d}> "


def format_source_text(
    text: str,
    task_prefix: str | None = DEFAULT_TASK_PREFIX,
    domain: str | None = None,
) -> str:
    domain_prefix = normalize_domain_tag(domain)
    prefix = normalize_task_prefix(task_prefix)
    return f"{domain_prefix}{prefix}{clean_text(text)}"


def format_source_batch(
    texts: list[str],
    task_prefix: str | None = DEFAULT_TASK_PREFIX,
    domains: list[str] | None = None,
) -> list[str]:
    if domains is None:
        return [format_source_text(t, task_prefix=task_prefix) for t in texts]
    return [
        format_source_text(t, task_prefix=task_prefix, domain=d)
        for t, d in zip(texts, domains)
    ]


def clean_target_batch(texts: list[str]) -> list[str]:
    return [clean_text(t) for t in texts]
