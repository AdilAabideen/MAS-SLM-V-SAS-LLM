"""Usage Extractor module helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class UsageExtractionResult:
    """Standardized provider token-usage extraction output."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    usage_source: Optional[str] = None

    @property
    def has_usage(self) -> bool:
        """True when any provider token field was present."""
        # Keep the main step clear.
        return (
            self.input_tokens is not None
            or self.output_tokens is not None
            or self.total_tokens is not None
        )


def _to_int_or_none(value: Any) -> Optional[int]:
    """Handle int or none."""
    # Keep the main step clear.
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _build_usage_result(
    *,
    input_tokens: Any,
    output_tokens: Any,
    total_tokens: Any,
    usage_source: str,
) -> UsageExtractionResult:
    """Build usage result."""
    # Build the next value.
    in_tok = _to_int_or_none(input_tokens)
    out_tok = _to_int_or_none(output_tokens)
    tot_tok = _to_int_or_none(total_tokens)

    if in_tok is None and out_tok is None and tot_tok is None:
        return UsageExtractionResult()

    if in_tok is None:
        in_tok = 0
    if out_tok is None:
        out_tok = 0
    if tot_tok is None:
        tot_tok = int(in_tok + out_tok)

    return UsageExtractionResult(
        input_tokens=max(0, int(in_tok)),
        output_tokens=max(0, int(out_tok)),
        total_tokens=max(0, int(tot_tok)),
        usage_source=usage_source,
    )


def extract_provider_usage(obj: Any) -> UsageExtractionResult:
    """
    Extract provider usage from common LangChain response fields.

    Parse priority:
    1) `usage_metadata` with `input_tokens`/`output_tokens`/`total_tokens`
    2) `response_metadata.token_usage` with OpenAI-style aliases:
       - `prompt_tokens` or `input_tokens`
       - `completion_tokens` or `output_tokens`
       - `total_tokens`
    """
    # Pull out the needed value.
    usage = getattr(obj, "usage_metadata", None)
    if isinstance(usage, Mapping):
        result = _build_usage_result(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=usage.get("total_tokens"),
            usage_source="provider_usage_metadata",
        )
        if result.has_usage:
            return result

    response_meta = getattr(obj, "response_metadata", None)
    if isinstance(response_meta, Mapping):
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, Mapping):
            result = _build_usage_result(
                input_tokens=token_usage.get("prompt_tokens", token_usage.get("input_tokens")),
                output_tokens=token_usage.get(
                    "completion_tokens",
                    token_usage.get("output_tokens"),
                ),
                total_tokens=token_usage.get("total_tokens"),
                usage_source="provider_response_metadata_token_usage",
            )
            if result.has_usage:
                return result

    return UsageExtractionResult()
