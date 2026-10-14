"""Bounded OpenRouter requests with usage accounting and no automatic paid retries."""

import json
import math
import time

import httpx


API_URL = 'https://openrouter.ai/api/v1'
# Planning allowance, checked against actual usage after every request.
# OpenRouter documents Parallel fast searches at $0.001 each (October 2026).
SEARCH_ALLOWANCE_USD = 0.001


class OpenRouterError(RuntimeError):
    """Safe HTTP error; never includes keys or raw provider responses."""

    def __init__(self, code, detail=''):
        self.code = code
        messages = {
            400: 'Request rejected. Check model support for tools and structured output.',
            401: 'Invalid OpenRouter API key.',
            402: 'OpenRouter credits are insufficient.',
            403: 'OpenRouter access denied; check key limits and provider permissions.',
            404: 'OpenRouter model unavailable. Check the selected model names.',
            429: 'OpenRouter rate or quota limit reached. Do not repeatedly resubmit.',
        }
        super().__init__(detail or messages.get(code, 'OpenRouter request failed. Try again later.'))


class BudgetExceeded(RuntimeError):
    """The application will not start another paid request."""


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


class OpenRouterClient:
    def __init__(self, settings, log):
        self.settings = settings
        self.log = log
        self.http = httpx.Client(timeout=httpx.Timeout(120, connect=15))
        self.prices = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.http.close()

    def _pricing(self, model):
        if not self.prices:
            # This metadata request is free; refuse paid work if pricing is unavailable.
            try:
                response = self.http.get(API_URL + '/models')
                response.raise_for_status()
                entries = response.json()['data']
                self.prices = {item['id']: item.get('pricing', {}) for item in entries}
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise BudgetExceeded('Cannot check current model pricing. No paid request started.') from None
        price = self.prices.get(model, {})
        rates = [_number(price.get(name)) for name in ('prompt', 'completion')]
        if any(rate is None for rate in rates):
            raise BudgetExceeded('Selected model has no usable published pricing. Check the model ID.')
        return rates

    def complete(self, stage, model, system, data, *, schema=None, search=False):
        if len(self.log.calls) >= self.settings.max_research_rounds * 3:
            raise BudgetExceeded('Maximum API calls reached; stopped further spending.')
        input_rate, output_rate = self._pricing(model)
        user_text = json.dumps(data, ensure_ascii=False)
        if len(user_text.encode()) > 45000:
            raise BudgetExceeded('Research context exceeded the size limit.')
        max_tokens = 1600 if stage == 'research' else 2000
        # Byte count intentionally overestimates common English token usage.
        estimated_input = len((system + user_text).encode()) + 1024 + (10000 if search else 0)
        reserved = estimated_input * input_rate + max_tokens * output_rate
        reserved += SEARCH_ALLOWANCE_USD if search else 0
        spent = self.log.summary()['estimated_budget_used_usd']
        if spent + reserved > self.settings.max_run_cost_usd:
            raise BudgetExceeded('Next request could exceed the configured run budget; stopped before calling the model.')
        payload = {
            'model': model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user_text}],
            'temperature': 0.1, 'stream': False, 'max_tokens': max_tokens,
            'usage': {'include': True},
            # A saved account web plugin must not search again during writing/review.
            'plugins': [{'id': 'web', 'enabled': False}],
            'provider': {'require_parameters': True,
                         'max_price': {'prompt': input_rate * 1000000,
                                       'completion': output_rate * 1000000}},
        }
        if schema:
            payload['response_format'] = {'type': 'json_schema',
                'json_schema': {'name': 'news_agent', 'strict': True, 'schema': schema}}
        if search:
            payload['tools'] = [{'type': 'openrouter:web_search', 'parameters': {
                'engine': 'parallel', 'mode': 'fast', 'max_results': 5,
                'max_total_results': 5, 'max_uses': 1, 'max_characters': 2000}}]
            # Allow the server loop to finish after one search; max_uses still
            # caps actual Parallel searches at one per request.
            payload['max_tool_calls'] = 2
        call = dict(stage=stage, model=model, reserved_usd=round(reserved, 8),
                    cost_usd=None, search_enabled=search, search_requests=None,
                    input_tokens=0, output_tokens=0, status='started')
        self.log.calls.append(call)
        self.log.stage = stage
        self.log.save()  # Persist the attempt before making a paid request.
        started = time.monotonic()
        try:
            response = self.http.post(API_URL + '/chat/completions',
                headers={'Authorization': f'Bearer {self.settings.api_key}'}, json=payload)
            try:
                body = response.json()
            except ValueError:
                body = {}
            usage = body.get('usage') or {} if isinstance(body, dict) else {}
            call['cost_usd'] = _number(usage.get('cost'))
            call['input_tokens'] = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
            call['output_tokens'] = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
            search_usage = (usage.get('server_tool_use') or {}).get('web_search_requests')
            call['search_requests'] = int(search_usage) if search_usage is not None else None
            if response.status_code >= 400:
                call['status'] = f'http_{response.status_code}'
                raise OpenRouterError(response.status_code)
            if not isinstance(body, dict) or body.get('error'):
                raise OpenRouterError(502)
            if call['search_requests'] and call['search_requests'] > int(search):
                call['status'] = 'unexpected_search_usage'
                raise BudgetExceeded('Provider reported unexpected extra searches; stopped further requests.')
            try:
                choice = body['choices'][0]
                message = choice['message']
                call['finish_reason'] = str(choice.get('finish_reason') or 'missing')[:40]
                call['pending_tool_names'] = [str(tool.get('function', {}).get('name', ''))[:80]
                                               for tool in message.get('tool_calls') or []
                                               if isinstance(tool, dict)]
                call['provider_request_id'] = str(body.get('id') or '')[:120]
                text = message['content']
                annotations = list(message.get('annotations') or [])
                if isinstance(text, list):
                    for part in text:
                        if isinstance(part, dict):
                            annotations.extend(part.get('annotations') or [])
                    text = ''.join(part.get('text', '') for part in text
                                   if isinstance(part, dict) and part.get('type') == 'text')
                if choice.get('finish_reason') != 'stop' or not isinstance(text, str) or not text.strip():
                    if not (search and annotations):
                        raise OpenRouterError(502, 'Provider returned an incomplete response '
                            f"({call['finish_reason']}). No automatic paid retry.")
            except (ValueError, KeyError, IndexError, TypeError):
                raise OpenRouterError(502) from None
            call['status'] = 'complete'
            return {'text': text, 'annotations': annotations}
        except httpx.HTTPError:
            call['status'] = 'connection_failed_cost_unknown'
            raise
        finally:
            call['elapsed_seconds'] = round(time.monotonic() - started, 2)
            if call['status'] == 'started':
                call['status'] = 'invalid_response'
            self.log.save()
