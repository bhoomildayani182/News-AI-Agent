"""Budgeted OpenRouter requests with one run-wide, length-only recovery allowance."""

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


class RunLimitReached(BudgetExceeded):
    """An expected, actionable limit stop, not an HTTP/server failure."""

    def __init__(self, message, *, kind='cost', setting='OPENROUTER_MAX_RUN_COST_USD'):
        super().__init__(message)
        self.kind, self.setting = kind, setting


class ProviderResponseError(RuntimeError):
    """An unusable provider response; never invent an HTTP status for it."""


class _TruncatedResponse(RuntimeError):
    pass


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


class OpenRouterClient:
    def __init__(self, settings, log, on_progress=None):
        self.settings = settings
        self.log = log
        self.http = httpx.Client(timeout=httpx.Timeout(120, connect=15))
        self.prices = {}
        self.models = {}
        self.on_progress = on_progress

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
                self.models = {item['id']: item for item in entries}
                self.prices = {item['id']: item.get('pricing', {}) for item in entries}
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                raise BudgetExceeded('Cannot check current model pricing. No paid request started.') from None
        price = self.prices.get(model, {})
        rates = [_number(price.get(name)) for name in ('prompt', 'completion')]
        if any(rate is None for rate in rates):
            raise BudgetExceeded('Selected model has no usable published pricing. Check the model ID.')
        return rates

    def complete(self, stage, model, system, data, *, schema=None, search=False):
        self._pricing(model)
        tokens = self.settings.research_max_tokens if search else self.settings.report_max_tokens
        model_cap = (self.models.get(model, {}).get('top_provider') or {}).get('max_completion_tokens')
        if isinstance(model_cap, int) and model_cap > 0:
            tokens = min(tokens, model_cap)
        try:
            return self._complete_once(stage, model, system, data, schema=schema,
                                       search=search, max_tokens=tokens)
        except _TruncatedResponse:
            recovery_tokens = self.settings.recovery_max_tokens
            if isinstance(model_cap, int) and model_cap > 0:
                recovery_tokens = min(recovery_tokens, model_cap)
            used = sum(bool(c.get('length_retry')) for c in self.log.calls)
            if used < self.settings.max_length_retries and recovery_tokens > tokens:
                message = (f'Recovering the {stage} response with a larger output allowance '
                           '(one budget-checked attempt).')
                self.log.decisions.append(dict(action='length_recovery', reason=message))
                if self.on_progress:
                    self.on_progress(message)
                hint = (' The prior response was cut off. Generate a NEW complete response '
                        'from the supplied evidence, not a continuation of partial JSON. Keep the '
                        'brief concise, assessment reasons under 20 words and omit repetition. '
                        'Never omit required fields, invent evidence or skip validation.')
                actual_input = self.log.calls[-1].get('input_tokens', 0)
                # Evidence/schema are unchanged. Reuse measured input usage rather
                # than charging the budget estimate for every UTF-8 byte as a token.
                retry_input = actual_input + len(hint.encode()) + 1024 if actual_input and not search else None
                try:
                    return self._complete_once(stage, model,
                        system + hint, data, schema=schema, search=search, max_tokens=recovery_tokens,
                        length_retry=True, estimated_input_override=retry_input)
                except _TruncatedResponse:
                    pass
            setting = 'OPENROUTER_RESEARCH_MAX_TOKENS' if search else 'OPENROUTER_REPORT_MAX_TOKENS'
            if isinstance(model_cap, int) and recovery_tokens <= tokens:
                setting = 'OPENROUTER_REVIEW_MODEL' if stage in {'review', 'repair'} else 'OPENROUTER_MODEL'
                advice = ('The model/provider advertises an output cap that prevents a larger retry. '
                          'Try a narrower question or select a model with a higher output allowance. ')
            elif not self.settings.max_length_retries:
                advice = ('Automatic length recovery is disabled. Increase ' + setting +
                          ' or enable OPENROUTER_MAX_LENGTH_RETRIES=1 in .env. ')
            elif self.log.calls[-1].get('length_retry'):
                setting = 'OPENROUTER_RECOVERY_MAX_TOKENS'
                advice = 'Increase OPENROUTER_RECOVERY_MAX_TOKENS in .env, or narrow the question. '
            else:
                advice = ('The run-wide recovery allowance was already used. Narrow the question, '
                          'or increase ' + setting + ' and the recovery ceiling in .env. ')
            raise RunLimitReached(
                f'Your agent reached its response-length limit during {stage}. No complete report '
                'is available. ' + advice + 'Restart the app after changes (recreate Docker). '
                'If the budget also blocks recovery, '
                'review OPENROUTER_MAX_RUN_COST_USD. API work already performed may be charged.',
                kind='output', setting=setting) from None

    def _complete_once(self, stage, model, system, data, *, schema=None, search=False,
                       max_tokens, length_retry=False, estimated_input_override=None):
        # One final evidence-only repair is allowed, never an extra web search.
        if stage == 'repair' and (search or (not length_retry and any(c['stage'] == 'repair' for c in self.log.calls))):
            raise RunLimitReached('The automatic repair allowance has been used. Try a narrower question.',
                                  kind='requests', setting='RESEARCH_MAX_ROUNDS')
        normal_calls = sum(c['stage'] != 'repair' and not c.get('length_retry') for c in self.log.calls)
        if (normal_calls >= self.settings.max_research_rounds * 3 and stage != 'repair' and not length_retry) or (
                len(self.log.calls) >= self.settings.max_research_rounds * 3 + 1 + self.settings.max_length_retries):
            raise RunLimitReached('Your agent reached its request allowance. Try a narrower question; '
                                  'review RESEARCH_MAX_ROUNDS (maximum 4) and the run budget.',
                                  kind='requests', setting='RESEARCH_MAX_ROUNDS')
        if search and sum(c['search_enabled'] for c in self.log.calls) >= self.settings.max_research_rounds:
            raise RunLimitReached('Your agent reached its search allowance. Try a narrower question; '
                                  'review RESEARCH_MAX_ROUNDS (maximum 4).',
                                  kind='research', setting='RESEARCH_MAX_ROUNDS')
        input_rate, output_rate = self._pricing(model)
        user_text = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        if len(user_text.encode()) > 45000:
            raise RunLimitReached('Your question produced more evidence than the agent can process '
                                  'in one run. Please narrow the topic.', kind='context', setting='')
        # Byte count intentionally overestimates common English token usage.
        estimated_input = len((system + user_text).encode()) + 1024 + (10000 if search else 0)
        if estimated_input_override is not None:
            estimated_input = estimated_input_override
        reserved = estimated_input * input_rate + max_tokens * output_rate
        reserved += SEARCH_ALLOWANCE_USD if search else 0
        spent = self.log.summary()['estimated_budget_used_usd']
        if spent + reserved > self.settings.max_run_cost_usd:
            raise RunLimitReached(f'Your agent reached its ${self.settings.max_run_cost_usd:.3f} run budget '
                                  f'before the next {stage} request could start safely. For another attempt, '
                                  'increase OPENROUTER_MAX_RUN_COST_USD in .env and restart the app '
                                  '(recreate Docker), or narrow the question. This can increase charges.')
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
        spec = self.models.get(model, {})
        reasoning = spec.get('reasoning') or {}
        effort = self.settings.reasoning_effort
        if effort != 'provider_default' and 'reasoning' in spec.get('supported_parameters', []):
            allowed = reasoning.get('supported_efforts')
            if (allowed and effort not in allowed) or (effort == 'none' and reasoning.get('mandatory')):
                raise BudgetExceeded('The model does not support the configured reasoning effort. '
                                     'Set OPENROUTER_REASONING_EFFORT=provider_default or a supported level.')
            payload['reasoning'] = {'effort': effort}
        if search:
            payload['tools'] = [{'type': 'openrouter:web_search', 'parameters': {
                'engine': 'parallel', 'mode': 'fast', 'max_results': 5,
                'max_total_results': 5, 'max_uses': 1, 'max_characters': 2000}}]
            # Enforce explicit publisher requests at the search engine, not only
            # in a prompt the model may rewrite into a broad query.
            domains = data.get('search_domains', []) if isinstance(data, dict) else []
            approved = {'reuters.com', 'abc.net.au', 'afr.com', 'rba.gov.au',
                        'finance.yahoo.com', 'investing.com', 'marketscreener.com'}
            if domains and isinstance(domains, list) and all(d in approved for d in domains):
                payload['tools'][0]['parameters']['allowed_domains'] = domains
            # Allow the server loop to finish after one search; max_uses still
            # caps actual Parallel searches at one per request.
            payload['max_tool_calls'] = 2
        call = dict(stage=stage, model=model, reserved_usd=round(reserved, 8),
                    cost_usd=None, search_enabled=search, search_requests=None,
                    input_tokens=0, output_tokens=0, reasoning_tokens=None, max_tokens=max_tokens,
                    reasoning_effort=payload.get('reasoning', {}).get('effort', 'provider_default'),
                    length_retry=length_retry, status='started')
        self.log.calls.append(call)
        self.log.stage = stage
        self.log.save()  # Persist the attempt before making a paid request.
        started = time.monotonic()
        try:
            response = self.http.post(API_URL + '/chat/completions',
                headers={'Authorization': f'Bearer {self.settings.api_key}'}, json=payload)
            call['http_status'] = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {}
            usage = body.get('usage') or {} if isinstance(body, dict) else {}
            call['cost_usd'] = _number(usage.get('cost'))
            call['input_tokens'] = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
            call['output_tokens'] = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
            call['reasoning_tokens'] = _number((usage.get('completion_tokens_details') or {}).get('reasoning_tokens'))
            search_usage = (usage.get('server_tool_use') or {}).get('web_search_requests')
            call['search_requests'] = int(search_usage) if search_usage is not None else None
            if response.status_code >= 400:
                call['status'] = f'http_{response.status_code}'
                raise OpenRouterError(response.status_code)
            if not isinstance(body, dict) or body.get('error'):
                raise ProviderResponseError('The AI provider returned an unusable response. Please try again later.')
            if call['search_requests'] and call['search_requests'] > int(search):
                call['status'] = 'unexpected_search_usage'
                raise BudgetExceeded('Provider reported unexpected extra searches; stopped further requests.')
            try:
                choice = body['choices'][0]
                message = choice['message']
                call['finish_reason'] = str(choice.get('finish_reason') or 'missing')[:40]
                call['native_finish_reason'] = str(choice.get('native_finish_reason') or '')[:60]
                call['response_model'] = str(body.get('model') or model)[:120]
                call['provider'] = str(body.get('provider') or '')[:80]
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
                        if call['finish_reason'] == 'length':
                            call['status'] = 'output_truncated'
                            raise _TruncatedResponse()
                        raise ProviderResponseError('The AI provider did not return a complete, usable response '
                                                    f"({call['finish_reason']}). No report was sent.")
            except (ValueError, KeyError, IndexError, TypeError):
                raise ProviderResponseError('The AI provider returned an invalid response. Please try again later.') from None
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
