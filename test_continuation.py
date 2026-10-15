"""Offline checks: approval, per-attempt settings, partial delivery and no extra spend."""

import unittest
from unittest.mock import Mock, patch

from config import Settings
from email_delivery import build_email
from openrouter_client import RunLimitReached
from stages import ResearchResult
from web_app import JobState, page


class ContinuationTests(unittest.TestCase):
    def settings(self):
        return Settings(api_key='fake', model='test', provider='openrouter')

    def limit(self, with_sources=True):
        error = RunLimitReached('Missing Reuters coverage. Research allowance reached.',
                                kind='cost', setting='OPENROUTER_MAX_RUN_COST_USD')
        error.checkpoint = {'steps': [], 'evidence': {}}
        error.resume_sources = ([{'id': 'S1', 'title': 'Official decision',
            'url': 'https://www.rba.gov.au/example', 'publisher': 'rba.gov.au',
            'published_at': '', 'excerpt': 'Collected evidence, not a verified conclusion.'}]
            if with_sources else [])
        error.resume_query = 'Reuters RBA decision official announcement'
        return error

    def run_limit(self, state, error, send_error=None):
        with patch('web_app.get_settings', return_value=self.settings()), \
             patch('web_app.validate_email_settings'), \
             patch('web_app.research_topic', side_effect=error) as research, \
             patch('web_app.record_delivery'), \
             patch('web_app.send_email', side_effect=send_error) as send:
            state._run('Compare RBA coverage')
        return research, send

    def test_limit_emails_only_labelled_reading_links_without_another_model_call(self):
        state = JobState()
        research, send = self.run_limit(state, self.limit())
        research.assert_called_once()
        result = send.call_args.args[1]
        self.assertTrue(result.run_summary['partial'])
        self.assertIn('not a verified answer', result.report_text)
        self.assertNotIn('Collected evidence', result.report_text)
        self.assertEqual(state.status, 'limit_reached')
        self.assertEqual(state.continuation['budget'], 0.06)
        self.assertIn('partial reading-links email was sent', state.message)
        with patch('email_delivery._load_email_env'):
            email = build_email('Topic', result, 'a@example.com', 'b@example.com')
        self.assertIn('[Partial research]', email['Subject'])
        self.assertIn('not a verified answer', email.get_body(preferencelist=('html',)).get_content())

    def test_no_sources_means_no_partial_email(self):
        state = JobState()
        _, send = self.run_limit(state, self.limit(False))
        send.assert_not_called()
        self.assertIn('No email was sent', state.message)

    def test_partial_delivery_failure_does_not_leave_job_running(self):
        state = JobState()
        self.run_limit(state, self.limit(), OSError('offline'))
        self.assertEqual(state.status, 'limit_reached')
        self.assertIn('delivery is unconfirmed', state.message)

    def test_approval_consumed_once_and_reuses_evidence(self):
        state = JobState()
        self.run_limit(state, self.limit())
        offer = state.continuation
        self.assertFalse(state.start('Different topic', continue_run=True))
        with patch('web_app.Thread') as worker:
            self.assertTrue(state.start('Compare RBA coverage', continue_run=True))
            worker.assert_called_once()
        self.assertIsNone(state.continuation)
        self.assertFalse(state.start('Compare RBA coverage', continue_run=True))
        settings = self.settings()
        with patch('web_app.get_settings', return_value=settings), \
             patch('web_app.validate_email_settings'), patch('web_app.record_delivery'), \
             patch('web_app.research_topic', return_value=ResearchResult('Report', [])) as research, \
             patch('web_app.send_email'):
            state._run('Compare RBA coverage')
        used = research.call_args.args[1]
        self.assertEqual(used.max_run_cost_usd, 0.06)
        self.assertEqual(settings.max_run_cost_usd, 0.03)
        self.assertEqual(research.call_args.kwargs['checkpoint'], offer['checkpoint'])
        self.assertTrue(research.call_args.kwargs['fresh'])
        with patch('web_app.Thread'):
            self.assertTrue(state.start('New ordinary topic'))
        self.assertIsNone(state.override)

    def test_continuation_failure_does_not_offer_unlimited_paid_attempts(self):
        state = JobState()
        self.run_limit(state, self.limit())
        with patch('web_app.Thread'):
            state.start('Compare RBA coverage', continue_run=True)
        self.run_limit(state, self.limit(False))
        self.assertIsNone(state.continuation)

    def test_gui_requires_explicit_approval_and_discloses_additional_charges(self):
        state = JobState()
        self.run_limit(state, self.limit(False))
        with patch('web_app.STATE', state):
            html = page('Compare RBA coverage', 'limit_reached').decode()
            self.assertIn('Approve additional budget and continue', html)
            self.assertIn('Previous charges still apply', html)
            self.assertNotIn('Approve additional budget', page('Different topic', 'limit_reached').decode())

    def test_missing_evidence_does_not_offer_a_pointless_budget_increase(self):
        state = JobState()
        error = self.limit()
        error.kind = 'research'
        self.run_limit(state, error)
        self.assertEqual(state.continuation['total_cap'], 0.03)
        self.assertEqual(state.continuation['rounds'], 4)
        self.assertIn('Increasing the money budget alone will not fix', state.message)
        with patch('web_app.STATE', state):
            html = page('Compare RBA coverage', 'limit_reached').decode()
        self.assertIn('Continue within existing budget', html)
        self.assertNotIn('Approve additional budget', html)

    def test_prior_spend_is_subtracted_from_approved_total_cap(self):
        state = JobState()
        error = self.limit(False)
        error.run_summary = {'estimated_budget_used_usd': 0.012882, 'reported_cost_usd': 0.012882}
        self.run_limit(state, error)
        self.assertAlmostEqual(state.continuation['budget'], 0.047118)
        self.assertEqual(state.continuation['total_cap'], 0.06)
        self.assertEqual(state.continuation['tokens'], 8192)
        self.assertEqual(state.continuation['report_tokens'], 4096)

    def test_extra_rounds_keep_total_money_cap_and_only_use_remaining_allowance(self):
        state = JobState()
        error = self.limit(False)
        error.kind = 'research'
        error.run_summary = {'estimated_budget_used_usd': 0.01, 'reported_cost_usd': 0.01}
        self.run_limit(state, error)
        self.assertEqual(state.continuation['rounds'], 4)
        self.assertAlmostEqual(state.continuation['budget'], 0.02)
        self.assertEqual(state.continuation['total_cap'], 0.03)
        self.assertEqual(self.settings().max_research_rounds, 2)

    def test_research_then_cost_approval_keeps_cumulative_spending_and_rounds(self):
        state = JobState()
        error = self.limit(False)
        error.kind = 'research'
        error.run_summary = {'reported_cost_usd': 0.01, 'estimated_budget_used_usd': 0.01}
        self.run_limit(state, error)
        with patch('web_app.Thread'):
            state.start('Compare RBA coverage', continue_run=True)
        second = self.limit(False)
        second.run_summary = {'reported_cost_usd': 0.005, 'estimated_budget_used_usd': 0.005}
        self.run_limit(state, second)
        self.assertEqual(state.continuation['rounds'], 4)
        self.assertEqual(state.continuation['used_kinds'], ['research', 'cost'])
        self.assertAlmostEqual(state.continuation['previous_cost'], 0.015)
        self.assertAlmostEqual(state.continuation['budget'], 0.045)
        self.assertEqual(state.continuation['total_cap'], 0.06)


if __name__ == '__main__':
    unittest.main()
