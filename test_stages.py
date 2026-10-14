"""Offline pipeline tests: no key, network calls or API costs."""

import contextlib
import io
import json
import re
import unittest
from unittest.mock import patch

from google.genai import types

from config import Settings
from stages import (HEADINGS, Evidence, ResearchError, ResearchResult, Source,
                    format_report, research_topic, validate_structure)


def response(text, *, grounded=False, finish=types.FinishReason.STOP):
    if text.startswith('## '):
        sections = re.split(r'^## (.+?)\s*$', text, flags=re.MULTILINE)
        text = json.dumps(dict(zip(sections[1::2], sections[2::2])))
    metadata = None
    if grounded:
        metadata = types.GroundingMetadata(
            grounding_chunks=[
                types.GroundingChunk(web=types.GroundingChunkWeb(
                    title='Official source', uri='https://example.org/decision')),
                types.GroundingChunk(web=types.GroundingChunkWeb(
                    title='Duplicate', uri='https://example.org/decision')),
                types.GroundingChunk(web=types.GroundingChunkWeb(
                    title='Independent source', uri='https://example.com/article')),
            ],
            grounding_supports=[types.GroundingSupport(
                segment=types.Segment(text='The decision was announced.'),
                grounding_chunk_indices=[0, 1])],
        )
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text=text)]),
        finish_reason=finish, grounding_metadata=metadata)])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.draft = '\n\n'.join('## ' + h + '\nSupported test claim [S1] [S2].' for h in HEADINGS)
        self.responses = [response('Plan'), response('Evidence', grounded=True),
                          response('Evaluation'), response('Comparison'),
                          response(self.draft), response('{"passed":true,"issues":[]}')]
        self.settings = Settings('test-key', 'test-model')

    def run_pipeline(self, *, thorough=True):
        with patch('stages.genai.Client') as constructor:
            client = constructor.return_value.__enter__.return_value
            client.models.generate_content.side_effect = self.responses
            with contextlib.redirect_stdout(io.StringIO()):
                result = research_topic('A test topic', self.settings, thorough=thorough)
            return result, client.models.generate_content.call_args_list, constructor

    def test_all_stages_and_evidence_handoff(self):
        result, calls, constructor = self.run_pipeline()
        self.assertEqual(len(calls), 6)
        self.assertEqual([bool(c.kwargs['config'].tools) for c in calls],
                         [False, True, False, False, False, False])
        self.assertEqual(json.loads(calls[1].kwargs['contents'])['plan'], 'Plan')
        evaluation_input = json.loads(calls[2].kwargs['contents'])
        self.assertEqual(evaluation_input['grounding_supports'][0]['source_ids'], ['S1'])
        self.assertEqual(json.loads(calls[3].kwargs['contents'])['evaluation'], 'Evaluation')
        self.assertEqual(json.loads(calls[4].kwargs['contents'])['comparison'], 'Comparison')
        self.assertEqual(json.loads(calls[5].kwargs['contents'])['draft'], self.draft)
        self.assertEqual(len(result.sources), 2)
        self.assertIn('https://example.org/decision', format_report(result))
        constructor.return_value.__exit__.assert_called_once()

    def test_missing_grounding_stops_before_evaluation(self):
        self.responses[1] = response('Ungrounded notes')
        with self.assertRaisesRegex(ResearchError, 'fewer than two'):
            self.run_pipeline()

    def test_one_distinct_source_rejected(self):
        single = response('Evidence', grounded=True)
        single.candidates[0].grounding_metadata.grounding_chunks = [
            single.candidates[0].grounding_metadata.grounding_chunks[0]
        ]
        self.responses[1] = single
        with self.assertRaisesRegex(ResearchError, 'fewer than two'):
            self.run_pipeline()

    def test_truncation_stops_pipeline(self):
        self.responses[0] = response('Partial', finish=types.FinishReason.MAX_TOKENS)
        with self.assertRaisesRegex(ResearchError, 'complete response'):
            self.run_pipeline()

    def test_unknown_citation_rejected(self):
        self.responses[4] = response(self.draft.replace('[S1]', '[S99]'))
        with self.assertRaisesRegex(ResearchError, 'source references'):
            self.run_pipeline()

    def test_empty_section_rejected(self):
        self.responses[4] = response(self.draft.replace('Supported test claim [S1] [S2].', '', 1))
        with self.assertRaisesRegex(ResearchError, 'incomplete report sections'):
            self.run_pipeline()

    def test_failed_or_malformed_verdict_never_publishes(self):
        for verdict in ('{"passed":false,"issues":["Unsupported claim"]}',
                        '{"passed":true,"issues":["Contradiction"]}',
                        '{"passed":"true","issues":[]}', 'not json'):
            with self.subTest(verdict=verdict):
                self.responses = self.responses[:5] + [response(verdict), response(self.draft),
                                                       response(verdict)]
                with self.assertRaises(ResearchError):
                    self.run_pipeline()

    def test_fast_mode_reviews_claims_without_another_search(self):
        self.responses = [response('Evidence', grounded=True), response(self.draft),
                          response('{"passed":true,"issues":[]}')]
        result, calls, _ = self.run_pipeline(thorough=False)
        self.assertEqual(len(calls), 3)
        self.assertTrue(calls[0].kwargs['config'].tools)
        self.assertFalse(calls[1].kwargs['config'].tools)
        self.assertFalse(calls[2].kwargs['config'].tools)
        audit = json.loads(calls[2].kwargs['contents'])
        self.assertEqual(audit['draft'], self.draft)
        self.assertEqual(audit['evidence']['notes'], 'Evidence')
        self.assertEqual(json.loads(calls[1].kwargs['contents'])['evidence']['sources'][0]['id'], 'S1')
        self.assertIn('https://example.org/decision', format_report(result))

    def test_fast_mode_blocks_failed_or_malformed_claim_audit(self):
        for verdict in ('{"passed":false,"issues":["Unsupported consensus claim"]}',
                        '{"passed":true,"issues":["Missing evidence"]}', 'invalid JSON'):
            with self.subTest(verdict=verdict):
                self.responses = [response('Evidence', grounded=True), response(self.draft),
                                  response(verdict), response(self.draft), response(verdict)]
                with self.assertRaises(ResearchError):
                    self.run_pipeline(thorough=False)

    def test_one_revision_is_reviewed_before_publication(self):
        corrected = self.draft.replace('Supported test claim', 'Corrected claim')
        self.responses = [response('Evidence', grounded=True), response(self.draft),
                          response('{"passed":false,"issues":["Missing assumptions"]}'),
                          response(corrected), response('{"passed":true,"issues":[]}')]
        result, calls, _ = self.run_pipeline(thorough=False)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result.report_text, corrected)
        self.assertEqual(json.loads(calls[-1].kwargs['contents'])['draft'], corrected)
        self.assertEqual(sum(bool(c.kwargs['config'].tools) for c in calls), 1)

    def test_fast_mode_rejects_invalid_citation(self):
        self.responses = [response('Evidence', grounded=True),
                          response(self.draft.replace('[S1]', '[S9]'))]
        with self.assertRaisesRegex(ResearchError, 'source references'):
            self.run_pipeline(thorough=False)

    def test_reviewer_model_only_used_for_final_call(self):
        self.settings = Settings('test-key', 'research-model', review_model='review-model')
        self.responses = [response('Evidence', grounded=True), response(self.draft),
                          response('{"passed":true,"issues":[]}')]
        _, calls, _ = self.run_pipeline(thorough=False)
        self.assertEqual([call.kwargs['model'] for call in calls],
                         ['research-model', 'research-model', 'review-model'])

    def test_oversized_report_is_rejected_before_review(self):
        self.responses = [response('Evidence', grounded=True),
                          response(self.draft + ' word' * 401)]
        with self.assertRaisesRegex(ResearchError, '300 words'):
            self.run_pipeline(thorough=False)

    def test_grouped_citations_render_every_cited_source(self):
        sources = [Source(f'Source {i}', f'https://example.com/{i}') for i in range(1, 7)]
        draft = self.draft.replace('[S1] [S2]', '[S1] [S2, S3] [S5] [S6]')
        validate_structure(Evidence('Evidence', sources, []), draft)
        rendered = format_report(ResearchResult(draft, sources))
        for display, i in enumerate((1, 2, 3, 5, 6), 1):
            self.assertIn(f'[S{display}] Source {i}\nhttps://example.com/{i}', rendered)
        self.assertNotIn('[S4] Source 4', rendered)

    def test_unknown_grouped_citation_rejected(self):
        bad_draft = self.draft.replace('[S1] [S2]', '[S1, S99] [S2]')
        with self.assertRaisesRegex(ResearchError, 'source references'):
            validate_structure(Evidence('Evidence', [Source('A', 'https://example.com/a'),
                                                     Source('B', 'https://example.com/b')], []),
                               bad_draft)

    def test_malformed_grouped_citation_rejected(self):
        bad_draft = self.draft.replace('[S1] [S2]', '[S1, S2, unknown] [S2]')
        with self.assertRaisesRegex(ResearchError, 'malformed source references'):
            validate_structure(Evidence('Evidence', [Source('A', 'https://example.com/a'),
                                                     Source('B', 'https://example.com/b')], []),
                               bad_draft)


if __name__ == '__main__':
    unittest.main()
