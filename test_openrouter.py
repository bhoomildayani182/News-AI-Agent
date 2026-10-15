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
from openrouter_client import OpenRouterError, ProviderResponseError, RunLimitReached
from research_agent import add_sources, run_openrouter, _accepted_report, initial_query, is_official_source
from email_format import plain_email, html_email
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
        fetcher = patch('research_agent.refresh_sources', return_value=[])
        fetcher.start()
        self.addCleanup(fetcher.stop)
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
        self.policy = dict(mode='multiple_sources', primary_source_id='',
                           reason='Compare relevant reporting from independent publishers.')
        self.enough = dict(sufficient=True, missing_information=[])
        self.draft = json.dumps(dict(action='write', reason='Enough evidence',
                                    next_query='', sections=self.sections,
                                    evidence_policy=self.policy, evidence_check=self.enough))
        self.passed = json.dumps(dict(passed=True, needs_research=False, issues=[],
                                      next_query='', sections=self.sections,
                                      question_coverage=dict(complete=True, missing_information=[]),
                                      evidence_policy=self.policy,
                                      source_checks=[dict(source_id=f'S{i}', same_story=True,
                                          reason='Covers the requested announcement and period.')
                                          for i in (1, 2)]))

    def run_agent(self, responses, *, settings=None, topic='Test topic', fresh=False, catalog_options=None):
        with patch('openrouter_client.httpx.Client') as constructor:
            http = constructor.return_value
            catalog = Mock()
            catalog.json.return_value = {'data': [
                {'id': 'google/gemini-2.5-flash-lite',
                 'pricing': {'prompt': '0.0000001', 'completion': '0.0000004'}},
                {'id': 'google/gemini-3.1-flash-lite',
                 'pricing': {'prompt': '0.00000025', 'completion': '0.0000015'}}]}
            for item in catalog.json.return_value['data']:
                item.update((catalog_options or {}).get(item['id'], {}))
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

    def official_fixture(self):
        annotations = [annotation('https://www.cyber.gov.au/publication/annual-cyber-threat-report-2024-2025',
            'The original ACSC report covers 2024-25. ACSC responded to over 1,200 incidents, '
            'up 11%, and received over 84,700 cybercrime reports. Identity fraud remained common.',
            'ACSC Annual Cyber Threat Report 2024-25')]
        sections = {h: 'The ACSC responded to over 1,200 incidents in 2024-25 [S1].' for h in HEADINGS}
        sections['What different sources are saying'] = 'This is an official-report summary; independent corroboration is not included [S1].'
        policy = dict(mode='official_primary', primary_source_id='S1',
                      reason='Summarises the original ACSC government report, not competing opinions.')
        draft = dict(action='write', reason='Official findings answer the question.',
                     next_query='', sections=sections, evidence_policy=policy, evidence_check=self.enough)
        verdict = json.loads(self.passed)
        verdict.update(sections=sections, evidence_policy=policy,
                       source_checks=[dict(source_id='S1', same_story=True,
                                           reason='Original issuer report for the requested period.')])
        return annotations, draft, verdict

    def test_original_government_report_succeeds_without_second_publisher(self):
        annotations, draft, verdict = self.official_fixture()
        replies = [completion('Official facts', annotations=annotations),
                   completion(json.dumps(draft)), completion(json.dumps(verdict))]
        result, http = self.run_agent(replies, topic='Main findings of the ACSC 2024-25 report')
        self.assertEqual(http.post.call_count, 3)
        self.assertEqual(result.run_summary['evidence_mode'], 'official_primary')
        self.assertTrue(result.run_summary['single_publisher'])
        self.assertIn('Independent corroboration is not included', plain_email(result))
        self.assertIn('Independent corroboration is not included', html_email('ACSC', result, 'News'))
        self.assertTrue(self.store.recent()[0]['single_publisher'])
        cached, http = self.run_agent([], topic='Main findings of the ACSC 2024-25 report')
        http.post.assert_not_called()
        self.assertIn('Independent corroboration is not included', plain_email(cached))

    def test_commercial_single_source_cannot_claim_official_exception(self):
        annotations, _, verdict = self.official_fixture()
        pool = []
        add_sources(pool, [annotation('https://commercial.example.com/report',
                           'A commercial article claims to summarise the government annual report.')], ())
        with self.assertRaisesRegex(ResearchError, 'original official source'):
            _accepted_report(verdict, pool)

    def test_official_vendor_documentation_can_support_single_source_brief(self):
        question = "What changed in Google Cloud's most recent Cloud Run announcement? Prefer official documentation."
        for host in ('cloud.google.com', 'docs.cloud.google.com'):
            with self.subTest(host=host):
                _, draft, verdict = self.official_fixture()
                annotations = [annotation(f'https://{host}/run/docs/release-notes',
                    'Official Cloud Run release notes identify a dated product change and its availability status.',
                    'Cloud Run release notes — Google Cloud')]
                sections = {h: 'Google Cloud release notes describe a dated Cloud Run product change [S1].'
                            for h in HEADINGS}
                sections['What different sources are saying'] = 'This brief describes Google Cloud\'s own announcement; independent corroboration is not included [S1].'
                policy = dict(mode='official_primary', primary_source_id='S1',
                              reason='First-party product release notes answer the announcement question.')
                draft.update(sections=sections, evidence_policy=policy)
                verdict.update(sections=sections, evidence_policy=policy)
                result, http = self.run_agent([
                    completion('Notes', annotations=annotations), completion(json.dumps(draft)),
                    completion(json.dumps(verdict))], topic=question, fresh=True)
                self.assertEqual(http.post.call_count, 3)
                self.assertEqual(result.run_summary['evidence_mode'], 'official_primary')
                self.assertIn('Independent corroboration is not included', plain_email(result))

    def test_known_vendor_hosts_do_not_allow_lookalikes_or_user_hosted_sites(self):
        for host in ('cloud.google.com', 'docs.cloud.google.com', 'developers.google.com',
                     'docs.aws.amazon.com', 'aws.amazon.com', 'learn.microsoft.com',
                     'azure.microsoft.com', 'cyber.gov.au'):
            self.assertTrue(is_official_source('https://' + host + '/docs'), host)
        for host in ('cloud.google.com.evil.example', 'evilcloud.google.com',
                     'docs-cloud.google.com', 'user.github.io', 'google.example.com',
                     'unrelated-commercial.example', 'googleusercontent.com'):
            self.assertFalse(is_official_source('https://' + host + '/docs'), host)

    def test_unknown_commercial_host_cannot_self_declare_official_vendor(self):
        _, _, verdict = self.official_fixture()
        pool = []
        add_sources(pool, [annotation('https://cloud.google.com.evil.example/release',
                    'Claims to be official Google Cloud documentation, but is on a lookalike domain.')], ())
        with self.assertRaisesRegex(ResearchError, 'not eligible'):
            _accepted_report(verdict, pool)

    def test_official_primary_id_must_exist_and_be_cited(self):
        annotations, _, verdict = self.official_fixture()
        pool = []
        add_sources(pool, annotations + self.annotations, ())
        verdict['evidence_policy']['primary_source_id'] = 'S99'
        with self.assertRaisesRegex(ResearchError, 'original official source'):
            _accepted_report(verdict, pool)
        verdict['evidence_policy']['primary_source_id'] = 'S2'
        with self.assertRaisesRegex(ResearchError, 'declared original official source'):
            _accepted_report(verdict, pool)

    def test_multi_source_comparison_still_requires_two_publishers(self):
        annotations, _, verdict = self.official_fixture()
        pool = []
        add_sources(pool, annotations, ())
        verdict['evidence_policy'] = self.policy
        with self.assertRaisesRegex(ResearchError, 'two source links'):
            _accepted_report(verdict, pool)

    def test_official_exception_does_not_bypass_question_coverage_or_relevance(self):
        annotations, _, verdict = self.official_fixture()
        pool = []
        add_sources(pool, annotations, ())
        verdict['question_coverage']['complete'] = False
        with self.assertRaisesRegex(ResearchError, 'question-coverage'):
            _accepted_report(verdict, pool)
        verdict['question_coverage']['complete'] = True
        verdict['source_checks'][0]['same_story'] = False
        with self.assertRaisesRegex(ResearchError, 'same-story'):
            _accepted_report(verdict, pool)

    def test_insufficient_evidence_skips_paid_review_before_targeted_search(self):
        bad = json.loads(self.draft)
        bad['evidence_check'] = dict(sufficient=False, missing_information=['Actual report findings'])
        bad['next_query'] = 'original report key findings statistics'
        result, http = self.run_agent(self.standard()[:1] + [completion(json.dumps(bad))] + self.standard())
        self.assertEqual(http.post.call_count, 5)
        self.assertEqual([c['stage'] for c in result.run_summary['calls']],
                         ['research', 'decision', 'research', 'decision', 'review'])
        self.assertIn('Actual report findings', result.run_summary['decisions'][0]['reason'])

    def test_report_mention_without_findings_never_becomes_a_successful_brief(self):
        annotations, draft, _ = self.official_fixture()
        draft.update(action='research', next_query='ACSC original report actual key findings',
                     reason='Only the existence of a report is mentioned.',
                     evidence_check=dict(sufficient=False, missing_information=['Main findings']))
        replies = [completion('Report exists', annotations=annotations), completion(json.dumps(draft))] * 2
        with self.assertRaisesRegex(RunLimitReached, 'Research limit reached'):
            self.run_agent(replies)
        self.assertEqual(self.store.recent()[0]['api_calls'], 4)

    def test_main_findings_search_targets_actual_report_results(self):
        question = 'Main findings of the latest ACSC threat report?'
        self.assertIn('statistics trends executive summary', initial_query(question))
        self.assertEqual(initial_query('Latest RBA decision?'), 'Latest RBA decision?')

    def test_official_report_still_obeys_length_limit_and_can_be_shortened(self):
        annotations, draft, verdict = self.official_fixture()
        oversized = json.loads(json.dumps(verdict))
        oversized['sections']['Summary'] += ' detail' * 310
        replies = [completion('Official facts', annotations=annotations),
                   completion(json.dumps(draft)), completion(json.dumps(oversized)),
                   completion(json.dumps(verdict))]
        result, http = self.run_agent(replies)
        self.assertLessEqual(len(result.report_text.split()), 300)
        self.assertEqual(http.post.call_count, 4)
        self.assertEqual(result.run_summary['calls'][-1]['stage'], 'repair')

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
        review_data = json.loads(calls[2]['messages'][1]['content'])
        self.assertRegex(review_data['today'], r'^\d{4}-\d{2}-\d{2}$')
        self.assertEqual([call['max_tokens'] for call in calls], [2400, 4096, 4096])
        self.assertTrue(all(call['plugins'][0]['enabled'] is False for call in calls))
        self.assertAlmostEqual(result.run_summary['reported_cost_usd'], 0.0015)
        self.assertEqual(result.sources[0].url, 'https://rba.gov.au/decision')
        self.assertEqual(self.store.recent()[0]['status'], 'complete')
        self.assertNotIn(self.settings.api_key, json.dumps(self.store.recent()))

    def test_duplicate_publishers_trigger_only_one_targeted_search(self):
        replies = [completion('First', annotations=self.annotations[1:]),
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
        with self.assertRaisesRegex(RunLimitReached, 'Research limit reached'):
            self.run_agent(replies)
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 6)
        self.assertEqual(log['status'], 'limit_reached')
        self.assertEqual(log['stage'], 'review')
        self.assertAlmostEqual(log['reported_cost_usd'], 0.003)

    def test_local_draft_citation_failure_is_repaired_in_review_without_extra_call(self):
        one = {h: 'Supported test claim [S1].' for h in HEADINGS}
        bad = json.dumps(dict(action='write', reason='', next_query='', sections=one,
                              evidence_policy=self.policy, evidence_check=self.enough))
        result, http = self.run_agent([completion('Notes', annotations=self.annotations),
                                      completion(bad), completion(self.passed)])
        self.assertEqual(http.post.call_count, 3)
        self.assertIn('[S2]', result.report_text)

    def test_cannot_publish_unknown_final_citation(self):
        bad = json.loads(self.passed)
        bad['sections']['Summary'] = 'Unknown claim [S99].'
        with self.assertRaisesRegex(ResearchError, 'source references'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(bad))] * 2)
        self.assertEqual(self.store.recent()[0]['status'], 'failed')

    def test_missing_excerpt_stops_before_draft_and_review(self):
        no_text = [dict(type='url_citation', url='https://rba.gov.au/a', title='RBA')]
        with self.assertRaisesRegex(RunLimitReached, 'Research limit reached'):
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

    def test_abs_report_cannot_pass_with_unrelated_uk_source(self):
        annotations = [
            annotation('https://abs.gov.au/cpi/aug-2026',
                       'Australian CPI rose 4.0% annually in August 2026, released 30 September 2026.', 'ABS CPI'),
            annotation('https://ons.gov.uk/cpi/august2026',
                       'UK consumer price inflation was 3.1% annually in August 2026, released 16 September.', 'ONS UK CPI')]
        verdict = json.loads(self.passed)
        verdict['source_checks'][1] = dict(source_id='S2', same_story=False,
                                         reason='UK figures do not cover the Australian ABS release.')
        with self.assertRaisesRegex(ResearchError, 'same-story relevance'):
            self.run_agent([completion('Notes', annotations=annotations), completion(self.draft),
                    completion(json.dumps(verdict)), completion(json.dumps(verdict))], topic='Latest ABS inflation release')
        self.assertEqual(self.store.recent()[0]['status'], 'failed')

    def test_missing_or_invalid_source_assessments_fail_closed(self):
        for checks in (None, [], [dict(source_id='S1', same_story=True, reason='Official')],
                       [dict(source_id='S1', same_story='true', reason='Official')],
                       [dict(source_id='S99', same_story=True, reason='Unknown')],
                       [dict(source_id=['S1'], same_story=True, reason='Invalid identifier')],
                       [dict(source_id='S1', same_story=True, reason='')],
                       [dict(source_id='S1', same_story=True, reason='A')] * 2):
            verdict = json.loads(self.passed)
            verdict['source_checks'] = checks
            with self.subTest(checks=checks), self.assertRaisesRegex(ResearchError, 'same-story'):
                self.run_agent(self.standard()[:2] + [completion(json.dumps(verdict))] * 2, fresh=True)

    def test_missing_final_check_is_automatically_repaired_without_search(self):
        bad = json.loads(self.passed)
        bad['source_checks'] = bad['source_checks'][:1]
        result, http = self.run_agent(self.standard()[:2] +
                                      [completion(json.dumps(bad)), completion(self.passed)])
        self.assertEqual(http.post.call_count, 4)
        self.assertEqual(result.run_summary['calls'][-1]['stage'], 'repair')
        self.assertNotIn('tools', http.post.call_args_list[-1].kwargs['json'])
        self.assertEqual(sum(c['search_enabled'] for c in result.run_summary['calls']), 1)
        self.assertTrue(any(d['action'] == 'repair' for d in result.run_summary['decisions']))
        self.assertEqual(result.run_summary['status'], 'complete')

    def test_final_round_can_recover_in_seventh_call(self):
        gap = json.loads(self.passed)
        gap.update(passed=False, needs_research=True, issues=['Need reasons'], next_query='RBA reasons')
        missing = json.loads(self.passed)
        missing['source_checks'] = []
        replies = self.standard()[:2] + [completion(json.dumps(gap))]
        replies += self.standard()[:2] + [completion(json.dumps(missing)), completion(self.passed)]
        result, http = self.run_agent(replies)
        self.assertEqual(http.post.call_count, 7)
        self.assertEqual(sum(c['search_enabled'] for c in result.run_summary['calls']), 2)
        self.assertEqual(result.run_summary['calls'][-1]['stage'], 'repair')

    def test_recovery_cannot_flip_an_irrelevant_source_to_positive(self):
        rejected = json.loads(self.passed)
        rejected['source_checks'][1]['same_story'] = False
        with self.assertRaisesRegex(ResearchError, 'already rejected'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(rejected)),
                                                  completion(self.passed)])
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 4)
        self.assertEqual(log['status'], 'failed')

    def test_recovery_can_replace_irrelevant_citation_with_existing_relevant_evidence(self):
        extra = annotation('https://reuters.com/rba',
                           'Reuters reports the same RBA announcement and explains the reasons behind this decision.')
        bad = json.loads(self.passed)
        bad['source_checks'][1]['same_story'] = False
        repaired = json.loads(self.passed)
        repaired['sections'] = {h: body.replace('S2', 'S3') for h, body in self.sections.items()}
        repaired['source_checks'][1]['source_id'] = 'S3'
        result, http = self.run_agent([
            completion('Notes', annotations=self.annotations + [extra]), completion(self.draft),
            completion(json.dumps(bad)), completion(json.dumps(repaired))])
        self.assertNotIn('[S2]', result.report_text)
        self.assertIn('[S3]', result.report_text)
        self.assertEqual(http.post.call_count, 4)

    def test_failed_repair_does_not_retry_again(self):
        bad = json.loads(self.passed)
        bad['source_checks'] = []
        with self.assertRaisesRegex(ResearchError, 'same-story'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(bad))] * 2)
        self.assertEqual(self.store.recent()[0]['api_calls'], 4)

    def test_incomplete_answer_cannot_pass_even_when_citations_are_valid(self):
        bad = json.loads(self.passed)
        bad['question_coverage'] = dict(complete=False, missing_information=['RBA stated reasons'])
        with self.assertRaisesRegex(ResearchError, 'question-coverage'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(bad))] * 2)

    def test_repair_can_request_targeted_research_for_missing_reasons(self):
        bad = json.loads(self.passed)
        bad['question_coverage'] = dict(complete=False, missing_information=['Reasons'])
        gap = dict(bad, passed=False, needs_research=True,
                   issues=['Need official stated reasons'], next_query='RBA statement stated reasons')
        replies = self.standard()[:2] + [completion(json.dumps(bad)), completion(json.dumps(gap))]
        result, http = self.run_agent(replies + self.standard())
        self.assertEqual(http.post.call_count, 7)
        self.assertEqual(result.run_summary['status'], 'complete')

    def test_budget_prevents_repair_before_any_extra_charge(self):
        bad = json.loads(self.passed)
        bad['source_checks'] = []
        replies = [completion('Notes', annotations=self.annotations, cost=0.005),
                   completion(self.draft, cost=0.005), completion(json.dumps(bad), cost=0.005)]
        with self.assertRaisesRegex(RunLimitReached, 'budget'):
            self.run_agent(replies, settings=replace(self.settings, max_run_cost_usd=0.018))
        self.assertEqual(self.store.recent()[0]['api_calls'], 3)

    def test_same_story_gap_gets_one_targeted_followup_within_existing_bounds(self):
        rejected = json.loads(self.passed)
        rejected.update(passed=False, needs_research=True,
                        issues=['ONS is about the UK, not this Australian release.'],
                        next_query='Australian ABS August 2026 CPI ABC Reuters analysis')
        result, http = self.run_agent(self.standard()[:2] + [completion(json.dumps(rejected))]
                                      + self.standard())
        self.assertEqual(http.post.call_count, 6)
        self.assertEqual(result.run_summary['status'], 'complete')
        self.assertIn('Australian ABS August 2026',
                      http.post.call_args_list[3].kwargs['json']['messages'][1]['content'])
        review_prompt = http.post.call_args_list[2].kwargs['json']['messages'][0]['content']
        self.assertIn('same_story', review_prompt)
        self.assertIn('headline versus underlying', review_prompt)

    def test_old_cache_without_relevance_checks_is_not_reused(self):
        self.run_agent(self.standard())
        with self.store.connect() as db:
            key, payload = db.execute('SELECT key, data FROM cache').fetchone()
            cached = json.loads(payload)
            cached.pop('source_checks')
            db.execute('UPDATE cache SET data=? WHERE key=?', (json.dumps(cached), key))
        result, http = self.run_agent(self.standard())
        self.assertFalse(result.run_summary['cache_hit'])
        self.assertEqual(http.post.call_count, 3)

    def test_budget_blocks_request_before_spending(self):
        with self.assertRaisesRegex(RunLimitReached, 'budget'):
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

    def test_explicit_latest_question_always_bypasses_cached_report(self):
        topic = 'What changed in the most recent product announcement?'
        self.run_agent(self.standard(), topic=topic)
        result, http = self.run_agent(self.standard(), topic=topic)
        self.assertFalse(result.run_summary['cache_hit'])
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

    def test_exhausted_length_recovery_is_a_limit_not_fake_http_502(self):
        with self.assertRaisesRegex(RunLimitReached, 'response-length limit'):
            self.run_agent([completion('Incomplete', finish='length')] * 2)
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 2)
        self.assertEqual(log['status'], 'limit_reached')
        self.assertEqual(log['limit_kind'], 'output')
        self.assertEqual(log['calls'][-1]['http_status'], 200)

    def test_truncated_draft_recovers_only_that_stage_without_research_again(self):
        replies = self.standard()[:1] + [completion('{"action":', finish='length')]
        replies += [completion(self.draft), completion(self.passed)]
        result, http = self.run_agent(replies)
        calls = [c.kwargs['json'] for c in http.post.call_args_list]
        self.assertEqual([c['stage'] for c in result.run_summary['calls']],
                         ['research', 'decision', 'decision', 'review'])
        self.assertEqual([c['max_tokens'] for c in calls], [2400, 4096, 8192, 4096])
        self.assertNotIn('tools', calls[2])
        self.assertEqual(sum(c['search_enabled'] for c in result.run_summary['calls']), 1)
        self.assertAlmostEqual(result.run_summary['reported_cost_usd'], 0.002)

    def test_truncated_reviewer_recovers_without_resending_or_redrafting(self):
        result, http = self.run_agent(self.standard()[:2] + [completion('', finish='length'),
                                                           completion(self.passed)])
        self.assertEqual([c['stage'] for c in result.run_summary['calls']],
                         ['research', 'decision', 'review', 'review'])
        self.assertEqual(http.post.call_count, 4)
        self.assertTrue(result.run_summary['calls'][-1]['length_retry'])

    def test_only_one_length_recovery_is_allowed_across_all_stages(self):
        replies = self.standard()[:1] + [completion('', finish='length'), completion(self.draft),
                                        completion('', finish='length')]
        with self.assertRaises(RunLimitReached):
            self.run_agent(replies)
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 4)
        self.assertEqual(sum(bool(c.get('length_retry')) for c in log['calls']), 1)

    def test_budget_blocks_length_recovery_before_another_paid_request(self):
        replies = self.standard()[:1] + [completion('', finish='length', cost=0.02)]
        with self.assertRaisesRegex(RunLimitReached, 'run budget'):
            self.run_agent(replies, settings=replace(self.settings, model='google/gemini-3.1-flash-lite'))
        log = self.store.recent()[0]
        self.assertEqual(log['api_calls'], 2)
        self.assertEqual(log['limit_kind'], 'cost')

    def test_retry_can_be_disabled_explicitly(self):
        with self.assertRaises(RunLimitReached):
            self.run_agent([completion('', finish='length')],
                           settings=replace(self.settings, max_length_retries=0))
        self.assertEqual(self.store.recent()[0]['api_calls'], 1)

    def test_truncated_search_with_evidence_does_not_repeat_search(self):
        replies = [completion('', annotations=self.annotations, finish='length')]
        result, http = self.run_agent(replies + self.standard()[1:])
        self.assertEqual(http.post.call_count, 3)
        self.assertFalse(any(c.get('length_retry') for c in result.run_summary['calls']))

    def test_truncated_search_without_evidence_uses_at_most_two_search_requests(self):
        replies = [completion('', finish='length')] + self.standard()
        result, http = self.run_agent(replies)
        self.assertEqual(sum(c['search_enabled'] for c in result.run_summary['calls']), 2)
        self.assertEqual(http.post.call_count, 4)

    def test_bad_json_response_is_not_reported_as_http_502_or_retried(self):
        bad = Mock(status_code=200)
        bad.json.return_value = {'usage': {'cost': 0.001}, 'choices': []}
        with self.assertRaises(ProviderResponseError) as caught:
            self.run_agent([bad])
        self.assertNotIsInstance(caught.exception, OpenRouterError)
        self.assertEqual(self.store.recent()[0]['api_calls'], 1)

    def test_actual_upstream_502_remains_an_http_error_without_retry(self):
        bad = Mock(status_code=502)
        bad.json.return_value = {'error': {'message': 'private provider response'}}
        with self.assertRaises(OpenRouterError) as caught:
            self.run_agent([bad])
        self.assertEqual(caught.exception.code, 502)
        self.assertEqual(self.store.recent()[0]['calls'][0]['http_status'], 502)

    def test_eight_call_bound_includes_two_rounds_repair_and_length_recovery(self):
        gap = json.loads(self.passed)
        gap.update(passed=False, needs_research=True, issues=['Need date'], next_query='official date')
        missing = json.loads(self.passed)
        missing['source_checks'] = []
        replies = self.standard()[:2] + [completion(json.dumps(gap))] + self.standard()[:2]
        replies += [completion('', finish='length'), completion(json.dumps(missing)), completion(self.passed)]
        result, http = self.run_agent(replies)
        self.assertEqual(http.post.call_count, 8)
        self.assertEqual(result.run_summary['status'], 'complete')
        self.assertEqual(sum(c['search_enabled'] for c in result.run_summary['calls']), 2)

    def test_reasoning_effort_and_usage_are_recorded_only_when_supported(self):
        last = completion(self.passed)
        last.json.return_value['usage']['completion_tokens_details'] = {'reasoning_tokens': 80}
        options = {self.settings.review_model: {'supported_parameters': ['reasoning'],
                    'reasoning': {'supported_efforts': ['low', 'high'], 'mandatory': True}}}
        result, http = self.run_agent(self.standard()[:2] + [last], catalog_options=options)
        calls = [c.kwargs['json'] for c in http.post.call_args_list]
        self.assertNotIn('reasoning', calls[0])
        self.assertEqual(calls[-1]['reasoning'], {'effort': 'low'})
        self.assertEqual(result.run_summary['calls'][-1]['reasoning_tokens'], 80)

    def test_model_advertised_output_cap_is_respected(self):
        options = {self.settings.model: {'top_provider': {'max_completion_tokens': 3000}}}
        _, http = self.run_agent(self.standard(), catalog_options=options)
        self.assertEqual(http.post.call_args_list[1].kwargs['json']['max_tokens'], 3000)

    def test_model_output_cap_notice_does_not_recommend_impossible_larger_retry(self):
        options = {self.settings.model: {'top_provider': {'max_completion_tokens': 4096}}}
        with self.assertRaisesRegex(RunLimitReached, 'advertises an output cap') as caught:
            self.run_agent(self.standard()[:1] + [completion('', finish='length')], catalog_options=options)
        self.assertEqual(caught.exception.setting, 'OPENROUTER_MODEL')
        self.assertEqual(self.store.recent()[0]['api_calls'], 2)

    def test_new_output_settings_are_validated_and_default_to_balanced_limits(self):
        env = {'AI_PROVIDER': 'openrouter', 'OPENROUTER_API_KEY': 'test-key',
               'OPENROUTER_RESEARCH_MAX_TOKENS': '2400', 'OPENROUTER_REPORT_MAX_TOKENS': '4096',
               'OPENROUTER_RECOVERY_MAX_TOKENS': '8192', 'OPENROUTER_MAX_LENGTH_RETRIES': '1',
               'OPENROUTER_REASONING_EFFORT': 'low'}
        with patch.dict(os.environ, env):
            settings = get_settings()
        self.assertEqual((settings.research_max_tokens, settings.report_max_tokens,
                          settings.recovery_max_tokens), (2400, 4096, 8192))
        for changes in ({'OPENROUTER_MAX_LENGTH_RETRIES': '10'},
                        {'OPENROUTER_RECOVERY_MAX_TOKENS': '1000'},
                        {'OPENROUTER_REPORT_MAX_TOKENS': '-1'},
                        {'OPENROUTER_REASONING_EFFORT': 'unlimited'}):
            with patch.dict(os.environ, dict(env, **changes)), self.assertRaises(ConfigurationError):
                get_settings()

    def test_history_transactions_close_database_connections(self):
        import sqlite3
        with self.store.connect() as db:
            db.execute('SELECT 1')
        with self.assertRaises(sqlite3.ProgrammingError):
            db.execute('SELECT 1')

    def test_search_evidence_remains_usable_when_server_tool_budget_ends(self):
        replies = [completion('', annotations=self.annotations, finish='tool_calls'),
                   completion(self.draft), completion(self.passed)]
        result, _ = self.run_agent(replies)
        self.assertEqual(result.run_summary['status'], 'complete')

    def test_uncited_section_is_never_published_even_if_reviewer_passes(self):
        verdict = json.loads(self.passed)
        verdict['sections']['Summary'] = 'Uncited news claim.'
        with self.assertRaisesRegex(ResearchError, 'supporting citation'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(verdict))] * 2)

    def test_illustrative_dollar_example_is_rejected(self):
        verdict = json.loads(self.passed)
        verdict['sections']['Why it matters'] = 'For instance, a loan may cost $85 more. [S1]'
        with self.assertRaisesRegex(ResearchError, 'monetary examples'):
            self.run_agent(self.standard()[:2] + [completion(json.dumps(verdict))] * 2)

    def test_actual_cost_over_budget_prevents_another_request(self):
        with self.assertRaisesRegex(RunLimitReached, 'budget'):
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
