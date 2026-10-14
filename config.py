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
        review_model = os.getenv('OPENROUTER_REVIEW_MODEL', 'google/gemini-3.1-flash-lite').strip() or model
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
        if not math.isfinite(budget) or not 0 < budget <= 1:
            raise ValueError()
        if not 1 <= rounds <= 2 or not 0 <= cache_minutes <= 60:
            raise ValueError()
    except ValueError:
        raise ConfigurationError('Use a positive budget up to $1, 1–2 research rounds, '
                                 'and 0–60 cache minutes.') from None
    return Settings(api_key=api_key, model=model, preferred_sources=preferred,
                    review_model=review_model, provider=provider,
                    max_run_cost_usd=budget, max_research_rounds=rounds,
                    cache_minutes=cache_minutes)
