"""Offline coverage of cost bounds, research decisions, sources and cached reports."""

import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import ConfigurationError, Settings, get_settings
from openrouter_client import OpenRouterError
from research_agent import add_sources, run_openrouter
from run_tracking import RunStore
from stages import HEADINGS, ResearchError, research_topic


def completion(content, *, annotations=None, finish='stop', cost=0.0005):
    response = Mock(status_code=200)
    response.json.return_value = {
        'usage': {'cost': cost, 'prompt_tokens': 120, 'completion_tokens': 100,
                  'server_tool_use': {'web_search_requests': 1 if annotations else 0}},
        'choices': [{'finish_reason': finish,
                     'message': {'content': content, 'annotations': annotations or []}}]}
    return response


def annotation(url, text, title='News'):
    return {'type': 'url_citation', 'url_citation':
            {'url': url, 'title': title, 'content': text}}


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = RunStore(Path(self.temp.name) / 'runs.sqlite3')
        self.settings = Settings('private-test-key', 'google/gemini-2.5-flash-lite',
                                 review_model='google/gemini-3.1-flash-lite',
                                 provider='openrouter')
        self.annotations = [
            annotation('https://rba.gov.au/decision?utm_source=test',
                       'The official statement announces the latest decision and explains its date and reasons.', 'RBA decision'),
            annotation('https://abc.net.au/article',
                       'Independent reporting discusses the practical effects and reports on the official announcement.', 'ABC report')]
        self.sections = {h: 'Supported test claim [S1] [S2].' for h in HEADINGS}
        self.draft = json.dumps(dict(action='write', reason='Enough evidence',
                                    next_query='', sections=self.sections))
        self.passed = json.dumps(dict(passed=True, needs_research=False, issues=[],
                                      next_query='', sections=self.sections))

    def run_agent(self, responses, *, settings=None, topic='Test topic', fresh=False):
        with patch('openrouter_client.httpx.Client') as constructor:
            http = constructor.return_value
            catalog = Mock()
            catalog.json.return_value = {'data': [
                {'id': 'google/gemini-2.5-flash-lite',
                 'pricing': {'prompt': '0.0000001', 'completion': '0.0000004'}},
                {'id': 'google/gemini-3.1-flash-lite',
                 'pricing': {'prompt': '0.00000025', 'completion': '0.0000015'}}]}
            http.get.return_value = catalog
            http.post.side_effect = responses
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_openrouter(topic, settings or self.settings,
                                        store=self.store, fresh=fresh)
            return result, http

    def standard(self):
        return [completion('Notes', annotations=self.annotations),
                completion(self.draft), completion(self.passed)]

    def test_config_selects_openrouter_without_gemini_key(self):
        env = {'AI_PROVIDER': 'openrouter', 'OPENROUTER_API_KEY': 'test-key',
               'OPENROUTER_MODEL': '', 'OPENROUTER_REVIEW_MODEL': '',
               'GEMINI_API_KEY': ''}
        with patch.dict(os.environ, env):
            settings = get_settings()
        self.assertEqual(settings.provider, 'openrouter')
        self.assertEqual(settings.model, 'google/gemini-3.1-flash-lite')
        self.assertEqual(settings.review_model, settings.model)

    def test_missing_key_and_invalid_limits_fail_before_api(self):
        for overrides in ({'OPENROUTER_API_KEY': ''},
                          {'OPENROUTER_API_KEY': 'key', 'OPENROUTER_MAX_RUN_COST_USD': 'nan'},
                          {'OPENROUTER_API_KEY': 'key', 'RESEARCH_MAX_ROUNDS': '20'}):
            with patch.dict(os.environ, dict(AI_PROVIDER='openrouter', **overrides)):
                with self.assertRaises(ConfigurationError):
                    get_settings()

    def test_dispatch_preserves_gemini_path(self):
        with patch('research_agent.run_openrouter', return_value='result') as run:
            self.assertEqual(research_topic('Test', self.settings), 'result')
        run.assert_called_once()

    def test_typical_run_is_three_bounded_calls_with_cost(self):
        result, http = self.run_agent(self.standard())
        calls = [item.kwargs['json'] for item in http.post.call_args_list]
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]['max_tool_calls'], 2)
        self.assertEqual(calls[0]['tools'][0]['parameters']['engine'], 'parallel')
        self.assertEqual(calls[0]['tools'][0]['parameters']['mode'], 'fast')
        self.assertEqual(calls[0]['tools'][0]['parameters']['max_uses'], 1)
        self.assertNotIn('tools', calls[1])
        self.assertNotIn('tools', calls[2])
        self.assertEqual(calls[2]['model'], self.settings.review_model)
        self.assertTrue(all(call['max_tokens'] <= 2000 for call in calls))
        self.assertTrue(all(call['plugins'][0]['enabled'] is False for call in calls))
        self.assertAlmostEqual(result.run_summary['reported_cost_usd'], 0.0015)
        self.assertEqual(result.sources[0].url, 'https://rba.gov.au/decision')
        self.assertEqual(self.store.recent()[0]['status'], 'complete')
        self.assertNotIn(self.settings.api_key, json.dumps(self.store.recent()))

    def test_duplicate_publishers_trigger_only_one_targeted_search(self):
        replies = [completion('First', annotations=self.annotations[:1]),
                   completion('Second', annotations=self.annotations),
                   completion(self.draft), completion(self.passed)]
        result, http = self.run_agent(replies)
        self.assertEqual(http.post.call_count, 4)
        self.assertEqual(result.run_summary['search_requests'], 2)
        self.assertEqual(len(result.sources), 2)

    def test_model_can_request_more_research_before_drafting(self):
        decision = json.dumps(dict(action='research', reason='Need effective date',
                                   next_query='official effective date', sections={h: '' for h in HEADINGS}))
        replies = [completion('First', annotations=self.annotations), completion(decision),
                   completion('Second', annotations=self.annotations),
                   completion(self.draft), completion(self.passed)]
        result, http = self.run_agent(replies)
        self.assertEqual(http.post.call_count, 5)
        second_search = http.post.call_args_list[2].kwargs['json']
        self.assertIn('official effective date', second_search['messages'][1]['content'])
        self.assertEqual(result.run_summary['decisions'][0]['action'], 'research')

    def test_reviewer_can_request_missing_evidence_but_no_unbounded_loop(self):
        reject = json.dumps(dict(passed=False, needs_research=True,
                                 issues=['Need date'], next_query='official date', sections=self.sections))
        replies = [completion('First', annotations=self.annotations), completion(self.draft),
                   completion(reject), completion('Second', annotations=self.annotations),
                   completion(self.draft), completion(reject)]
        with self.assertRaisesRegex(ResearchError, 'Research limit reached'):
            self.run_agent(replies)
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 6)
        self.assertEqual(log['status'], 'failed')
        self.assertEqual(log['stage'], 'review')
        self.assertAlmostEqual(log['reported_cost_usd'], 0.003)

    def test_local_draft_citation_failure_is_repaired_in_review_without_extra_call(self):
        one = {h: 'Supported test claim [S1].' for h in HEADINGS}
        bad = json.dumps(dict(action='write', reason='', next_query='', sections=one))
        result, http = self.run_agent([completion('Notes', annotations=self.annotations),
                                      completion(bad), completion(self.passed)])
        self.assertEqual(http.post.call_count, 3)
        self.assertIn('[S2]', result.report_text)

    def test_cannot_publish_unknown_final_citation(self):
        bad = json.loads(self.passed)
        bad['sections']['Summary'] = 'Unknown claim [S99].'
        with self.assertRaisesRegex(ResearchError, 'source references'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(bad))])
        self.assertEqual(self.store.recent()[0]['status'], 'failed')

    def test_missing_excerpt_stops_before_draft_and_review(self):
        no_text = [dict(type='url_citation', url='https://rba.gov.au/a', title='RBA')]
        with self.assertRaisesRegex(ResearchError, 'Research limit reached'):
            self.run_agent([completion('Notes', annotations=no_text)] * 2)
        self.assertEqual(self.store.recent()[0]['api_calls'], 2)

    def test_source_normalization_and_syndication_dedup(self):
        pool = []
        shared = 'This identical syndicated excerpt has been published by multiple sites without new reporting.'
        add_sources(pool, [annotation('https://abc.net.au/a?utm_medium=x#top', shared),
                           annotation('https://abc.net.au/a', shared),
                           annotation('https://copy.example.org/a', shared),
                           annotation('javascript:alert(1)', shared)], ())
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0]['publisher'], 'abc.net.au')

    def test_budget_blocks_request_before_spending(self):
        with self.assertRaisesRegex(ResearchError, 'budget'):
            self.run_agent([], settings=replace(self.settings, max_run_cost_usd=0.000001))
        self.assertEqual(self.store.recent()[0]['api_calls'], 0)

    def test_cache_avoids_all_provider_calls_and_fresh_bypasses_it(self):
        self.run_agent(self.standard())
        cached, http = self.run_agent([])
        self.assertTrue(cached.run_summary['cache_hit'])
        http.post.assert_not_called()
        http.get.assert_not_called()
        self.assertEqual(cached.run_summary['reported_cost_usd'], 0)
        _, http = self.run_agent(self.standard(), fresh=True)
        self.assertEqual(http.post.call_count, 3)

    def test_402_has_no_paid_retry_and_cost_is_not_claimed_zero(self):
        response = Mock(status_code=402)
        response.json.return_value = {'error': {'message': 'private text never logged'}}
        with self.assertRaises(OpenRouterError):
            self.run_agent([response])
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 1)
        self.assertFalse(log['cost_complete'])
        self.assertNotIn('private text', json.dumps(log))

    def test_truncated_or_malformed_response_has_no_retry(self):
        with self.assertRaises(OpenRouterError):
            self.run_agent([completion('Incomplete', finish='length')])
        self.assertEqual(self.store.recent()[0]['api_calls'], 1)

    def test_search_evidence_remains_usable_when_server_tool_budget_ends(self):
        replies = [completion('', annotations=self.annotations, finish='tool_calls'),
                   completion(self.draft), completion(self.passed)]
        result, _ = self.run_agent(replies)
        self.assertEqual(result.run_summary['status'], 'complete')

    def test_uncited_section_is_never_published_even_if_reviewer_passes(self):
        verdict = json.loads(self.passed)
        verdict['sections']['Summary'] = 'Uncited news claim.'
        with self.assertRaisesRegex(ResearchError, 'supporting citation'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(verdict))])

    def test_illustrative_dollar_example_is_rejected(self):
        verdict = json.loads(self.passed)
        verdict['sections']['Why it matters'] = 'For instance, a loan may cost $85 more. [S1]'
        with self.assertRaisesRegex(ResearchError, 'monetary examples'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(verdict))])

    def test_actual_cost_over_budget_prevents_another_request(self):
        with self.assertRaisesRegex(ResearchError, 'budget'):
            self.run_agent([completion('Notes', annotations=self.annotations, cost=0.04)])
        self.assertEqual(self.store.recent()[0]['api_calls'], 1)

    def test_changed_preferences_do_not_reuse_cached_report(self):
        self.run_agent(self.standard())
        result, http = self.run_agent(self.standard(), settings=replace(
            self.settings, preferred_sources=('Reuters',)))
        self.assertFalse(result.run_summary['cache_hit'])
        self.assertEqual(http.post.call_count, 3)

    def test_unwritable_history_prevents_all_provider_calls(self):
        with patch.object(self.store, 'save_run', side_effect=ConfigurationError('Not writable')), \
             patch('research_agent.OpenRouterClient') as client:
            with self.assertRaises(ConfigurationError):
                run_openrouter('Test', self.settings, store=self.store)
        client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
