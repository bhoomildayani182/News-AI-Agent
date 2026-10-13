"""Small OpenRouter adapter for the existing research pipeline."""

from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
from google.genai import types


API_URL = 'https://openrouter.ai/api/v1/chat/completions'


class OpenRouterError(RuntimeError):
    """Safe HTTP error; never includes the API key or raw provider response."""

    def __init__(self, code):
        self.code = code
        messages = {
            400: 'Request rejected. Check model support for web search and structured output.',
            401: 'Invalid OpenRouter API key.',
            402: 'OpenRouter credits are insufficient.',
            403: 'OpenRouter access denied.',
            404: 'OpenRouter model unavailable. Check OPENROUTER_MODEL.',
            429: 'OpenRouter rate or quota limit reached.',
        }
        super().__init__(messages.get(code, 'OpenRouter request failed. Try again later.'))


def _json_schema(schema):
    """Convert the pipeline's simple Gemini schema to standard JSON Schema."""
    def convert(node):
        result = {'type': node['type'].lower()}
        if 'properties' in node:
            result['properties'] = {name: convert(value)
                                    for name, value in node['properties'].items()}
            result['required'] = node.get('required', list(node['properties']))
            result['additionalProperties'] = False
        if 'items' in node:
            result['items'] = convert(node['items'])
        return result
    return convert(schema)


def _annotations(message):
    """Keep only source URLs accompanied by a provider supplied excerpt."""
    sources, supports, seen = [], [], set()
    for item in message.get('annotations') or []:
        if not isinstance(item, dict) or item.get('type') != 'url_citation':
            continue
        citation = item.get('url_citation') or item
        if not isinstance(citation, dict):
            continue
        excerpt = str(citation.get('content') or '').strip()
        if not excerpt:
            # A link alone cannot verify a claim in the later review stage.
            continue
        url = str(citation.get('url') or '').strip()
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            continue
        if url not in seen:
            seen.add(url)
            title = str(citation.get('title') or parsed.netloc).strip()
            sources.append(SimpleNamespace(web=SimpleNamespace(uri=url, title=title)))
        index = next(i for i, source in enumerate(sources) if source.web.uri == url)
        supports.append(SimpleNamespace(
            segment=SimpleNamespace(text=excerpt[:1200]),
            grounding_chunk_indices=[index],
        ))
    return SimpleNamespace(grounding_chunks=sources, grounding_supports=supports)


class OpenRouterClient:
    """Expose the few methods used by stages.py without another SDK dependency."""

    def __init__(self, api_key):
        self.http = httpx.Client(timeout=120)
        self.api_key = api_key
        self.models = self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def generate_content(self, *, model, config, contents):
        search = bool(config.tools)
        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': config.system_instruction},
                {'role': 'user', 'content': contents},
            ],
            'temperature': 0.2,
            'stream': False,
        }
        if search:
            payload['tools'] = [{
                'type': 'openrouter:web_search',
                # Exa returns extractive highlights in url_citation.content.
                # Native search can provide links without reviewable excerpts.
                'parameters': {'engine': 'exa', 'max_results': 5,
                               'max_total_results': 10, 'max_uses': 2,
                               'max_characters': 2000},
            }]
            payload['max_tool_calls'] = 2
        if config.response_schema:
            payload['response_format'] = {
                'type': 'json_schema',
                'json_schema': {'name': 'news_agent_response', 'strict': True,
                                'schema': _json_schema(config.response_schema)},
            }
            payload['provider'] = {'require_parameters': True}
        response = self.http.post(
            API_URL,
            headers={'Authorization': f'Bearer {self.api_key}',
                     'Content-Type': 'application/json'},
            json=payload,
        )
        if response.status_code >= 400:
            raise OpenRouterError(response.status_code)
        try:
            choice = response.json()['choices'][0]
            message = choice['message']
            content = message['content']
            annotations = list(message.get('annotations') or [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        annotations.extend(part.get('annotations') or [])
                content = ''.join(part.get('text', '') for part in content
                                  if isinstance(part, dict) and part.get('type') == 'text')
            if not isinstance(content, str):
                content = ''
            metadata = _annotations({'annotations': annotations}) if search else None
            finish = types.FinishReason.STOP if choice.get('finish_reason') == 'stop' else types.FinishReason.MAX_TOKENS
        except (KeyError, IndexError, TypeError, ValueError):
            raise OpenRouterError(502) from None
        return SimpleNamespace(
            text=content,
            candidates=[SimpleNamespace(finish_reason=finish, grounding_metadata=metadata)],
        )
