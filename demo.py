"""Clearly fictional sample data for zero-cost, offline formatting tests."""

from stages import ResearchResult, Source


DEMO_TOPIC = 'Example news topic (fictional test only)'


def demo_result() -> ResearchResult:
    return ResearchResult(
        report_text='''## Summary
This is a fictional sample used to test the program. [S1]

## In simple terms
This section shows how the agent explains specialist words plainly. [S1]

## Why it matters
The sample change could affect a pretend household budget. [S1]

## What different sources are saying
Two fictional sources describe the same event from different angles. [S1] [S2]

## What to watch next
Watch for the next fictional announcement. [S2]''',
        sources=[
            Source('Fictional official source', 'https://example.org/official'),
            Source('Fictional independent source', 'https://example.com/article'),
        ],
    )
