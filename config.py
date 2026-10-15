"""Application configuration loaded from environment variables."""

from dataclasses import dataclass, field
import math
import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_OPENROUTER_MODEL = 'google/gemini-3.1-flash-lite'


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


@dataclass(frozen=True)
class Settings:
    """Configuration values needed by the selected AI provider."""

    api_key: str = field(repr=False)
    model: str
    preferred_sources: tuple[str, ...] = ()
    review_model: str = ''
    provider: str = 'gemini'
    max_run_cost_usd: float = 0.03
    max_research_rounds: int = 2
    cache_minutes: int = 15
    research_max_tokens: int = 2400
    report_max_tokens: int = 4096
    recovery_max_tokens: int = 8192
    max_length_retries: int = 1
    reasoning_effort: str = 'low'


def get_settings(*, require_api_key: bool = True) -> Settings:
    """Load settings without ever printing or logging the API key."""

    # load_dotenv reads a local .env file when present. The file is excluded
    # from Git so that a developer's API key is not committed accidentally.
    load_dotenv(Path(__file__).resolve().parent / '.env', override=False)

    provider = os.getenv('AI_PROVIDER', 'gemini').strip().lower() or 'gemini'
    if provider not in {'gemini', 'openrouter'}:
        raise ConfigurationError('AI_PROVIDER must be gemini or openrouter.')
    key_name = 'OPENROUTER_API_KEY' if provider == 'openrouter' else 'GEMINI_API_KEY'
    api_key = os.getenv(key_name, '').strip()
    if require_api_key and (
        not api_key or api_key in {'replace_with_your_api_key', 'your_real_api_key_here',
                                   'your_openrouter_key_here', 'your_real_openrouter_key_here'}
    ):
        raise ConfigurationError(f'{key_name} is not set in .env.')
    if provider == 'openrouter':
        model = os.getenv('OPENROUTER_MODEL', DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
        review_model = os.getenv('OPENROUTER_REVIEW_MODEL', 'google/gemini-3-flash-preview').strip() or model
    else:
        model = os.getenv('GEMINI_MODEL', DEFAULT_MODEL).strip() or DEFAULT_MODEL
        review_model = os.getenv('GEMINI_REVIEW_MODEL', 'gemini-3.7-flash').strip() or model
    preferred = tuple(dict.fromkeys(
        source.strip() for source in os.getenv('PREFERRED_SOURCES', 'RBA,ABC,Reuters,AFR').split(',')
        if source.strip()
    ))
    try:
        budget = float(os.getenv('OPENROUTER_MAX_RUN_COST_USD', '0.03'))
        rounds = int(os.getenv('RESEARCH_MAX_ROUNDS', '2'))
        cache_minutes = int(os.getenv('REPORT_CACHE_MINUTES', '15'))
        research_tokens = int(os.getenv('OPENROUTER_RESEARCH_MAX_TOKENS', '2400'))
        report_tokens = int(os.getenv('OPENROUTER_REPORT_MAX_TOKENS', '4096'))
        recovery_tokens = int(os.getenv('OPENROUTER_RECOVERY_MAX_TOKENS', '8192'))
        length_retries = int(os.getenv('OPENROUTER_MAX_LENGTH_RETRIES', '1'))
        reasoning_effort = os.getenv('OPENROUTER_REASONING_EFFORT', 'low').strip().lower() or 'low'
        if not math.isfinite(budget) or not 0 < budget <= 1:
            raise ValueError()
        if not 1 <= rounds <= 4 or not 0 <= cache_minutes <= 60:
            raise ValueError()
        if (not 256 <= research_tokens <= 16384 or not 256 <= report_tokens <= 16384
                or not max(research_tokens, report_tokens) <= recovery_tokens <= 32768
                or length_retries not in (0, 1)
                or reasoning_effort not in {'provider_default', 'none', 'minimal', 'low', 'medium', 'high'}):
            raise ValueError()
    except ValueError:
        raise ConfigurationError('Use a positive budget up to $1, 1–4 research rounds, '
                                 'and 0–60 cache minutes. Output limits must be 256–16384 tokens; '
                                 'recovery must cover both output limits, up to 32768 tokens. '
                                 'Use 0 or 1 length retries and a valid reasoning effort.') from None
    return Settings(api_key=api_key, model=model, preferred_sources=preferred,
                    review_model=review_model, provider=provider,
                    max_run_cost_usd=budget, max_research_rounds=rounds,
                    cache_minutes=cache_minutes, research_max_tokens=research_tokens,
                    report_max_tokens=report_tokens, recovery_max_tokens=recovery_tokens,
                    max_length_retries=length_retries, reasoning_effort=reasoning_effort)
