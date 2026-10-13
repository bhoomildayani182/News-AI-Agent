"""Offline checks for OpenRouter provider selection and cited research."""

import contextlib
import io
import json
import os
import unittest
from unittest.mock import Mock, patch

from config import ConfigurationError, Settings, get_settings
from openrouter_client import OpenRouterError
from stages import HEADINGS, ResearchError, research_topic


def completion(content, *, annotations=None, finish='stop'):
    response = Mock(status_code=200)
    response.json.return_value = {'choices': [{
        'finish_reason': finish,
        'message': {'content': content, 'annotations': annotations or []},
    }]}
    return response


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings('test-openrouter-key', 'google/gemini-3.1-flash-lite',
                                 review_model='google/gemini-3.7-flash', provider='openrouter')
        self.annotations = [
            {'type': 'url_citation', 'url_citation': {
                'url': 'https://example.org/official', 'title': 'Official',
                'content': 'The decision was published today.'}},
            {'type': 'url_citation', 'url_citation': {
                'url': 'https://example.com/report', 'title': 'Independent',
                'content': 'The report explains the decision.'}},
        ]
        self.draft = json.dumps({heading: 'Supported claim [S1] [S2].'
                                 for heading in HEADINGS})

    def test_config_selects_openrouter_without_gemini_key(self):
        env = {'AI_PROVIDER': 'openrouter', 'OPENROUTER_API_KEY': 'test-key',
               'OPENROUTER_MODEL': 'google/gemini-3.1-flash-lite',
               'OPENROUTER_REVIEW_MODEL': 'google/gemini-3.7-flash',
               'GEMINI_API_KEY': ''}
        with patch.dict(os.environ, env):
            settings = get_settings()
        self.assertEqual(settings.provider, 'openrouter')
        self.assertEqual(settings.review_model, 'google/gemini-3.7-flash')

    def test_missing_openrouter_key_stops_before_api_call(self):
        with patch.dict(os.environ, {'AI_PROVIDER': 'openrouter',
                                     'OPENROUTER_API_KEY': ''}):
            with self.assertRaisesRegex(ConfigurationError, 'OPENROUTER_API_KEY'):
                get_settings()

    def test_fast_pipeline_searches_and_uses_provider_citations(self):
        replies = [completion('Evidence', annotations=self.annotations),
                   completion(self.draft),
                   completion('{"passed":true,"issues":[]}')]
        with patch('openrouter_client.httpx.Client') as constructor:
            constructor.return_value.post.side_effect = replies
            with contextlib.redirect_stdout(io.StringIO()):
                result = research_topic('Test topic', self.settings)
            calls = constructor.return_value.post.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(result.sources[0].url, 'https://example.org/official')
        self.assertIn('openrouter:web_search', str(calls[0].kwargs['json']['tools']))
        self.assertEqual(calls[0].kwargs['json']['tools'][0]['parameters']['engine'], 'exa')
        self.assertNotIn('tools', calls[1].kwargs['json'])
        self.assertEqual(calls[2].kwargs['json']['model'], 'google/gemini-3.7-flash')
        self.assertEqual(calls[1].kwargs['json']['response_format']['type'], 'json_schema')

    def test_search_without_provider_citations_stops(self):
        with patch('openrouter_client.httpx.Client') as constructor:
            constructor.return_value.post.return_value = completion('Ungrounded')
            with self.assertRaisesRegex(ResearchError, 'fewer than two'):
                research_topic('Test topic', self.settings)
            self.assertEqual(constructor.return_value.post.call_count, 1)

    def test_url_without_excerpt_cannot_be_used_as_evidence(self):
        annotations = [self.annotations[0],
                       {'type': 'url_citation', 'url_citation': {
                           'url': 'https://example.com/unread', 'title': 'Unread'}}]
        with patch('openrouter_client.httpx.Client') as constructor:
            constructor.return_value.post.return_value = completion('Evidence', annotations=annotations)
            with self.assertRaisesRegex(ResearchError, 'fewer than two'):
                research_topic('Test topic', self.settings)
            self.assertEqual(constructor.return_value.post.call_count, 1)

    def test_insufficient_credits_reports_safe_error(self):
        response = Mock(status_code=402)
        with patch('openrouter_client.httpx.Client') as constructor:
            constructor.return_value.post.return_value = response
            with self.assertRaises(OpenRouterError) as caught:
                research_topic('Test topic', self.settings)
        self.assertEqual(caught.exception.code, 402)
        self.assertNotIn(self.settings.api_key, str(caught.exception))


if __name__ == '__main__':
    unittest.main()
