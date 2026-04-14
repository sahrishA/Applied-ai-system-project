"""
A lightweight evaluation harness for DocuBot.

This module helps students compare:
- naive generation over the full docs
- retrieval only answers
- RAG answers (retrieval + Gemini)

The evaluation is intentionally simple: it checks whether DocuBot retrieves
the correct files for each query and reports a hit rate.
"""

import re

from dataset import SAMPLE_QUERIES


# -----------------------------------------------------------
# Expected document signals for evaluation
# -----------------------------------------------------------
# This dictionary maps a query substring to the filename(s)
# that should be relevant. It does NOT need to be perfect.
# It simply gives students a way to measure improvements.
#
# Example:
#   If a query contains the phrase "auth token",
#   evaluation expects AUTH.md to appear in the retrieval results.
#
EXPECTED_SOURCES = {
    "auth token": ["AUTH.md"],
    "environment variables": ["AUTH.md"],
    "database": ["DATABASE.md"],
    "users": ["API_REFERENCE.md"],
    "projects": ["API_REFERENCE.md"],
    "refresh": ["AUTH.md"],
    "users table": ["DATABASE.md"],
}


def expected_files_for_query(query):
    """
    Returns a list of expected filenames based on simple substring matching.
    """
    query_lower = query.lower()
    matches = []
    for key, files in EXPECTED_SOURCES.items():
        if key in query_lower:
            matches.extend(files)
    return matches


# -----------------------------------------------------------
# Evidence validation helpers


def _query_evidence_matches(query, text):
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    return bool(query_words & text_words)


def evaluate_retrieval_with_evidence(bot, top_k=3):
    """
    Runs DocuBot retrieval and validates that retrieved snippets also
    contain evidence from the query.

    Returns a tuple: (hit_rate, evidence_rate, detailed_results)
    """
    results = []
    hits = 0
    evidence_hits = 0

    for query in SAMPLE_QUERIES:
        expected = expected_files_for_query(query)
        retrieved = bot.retrieve(query, top_k=top_k)

        retrieved_files = [fname for fname, _ in retrieved]
        evidence = any(
            _query_evidence_matches(query, snippet_text)
            for _, snippet_text in retrieved
        )

        hit = any(f in retrieved_files for f in expected) if expected else False
        valid = hit and evidence
        if hit:
            hits += 1
        if valid:
            evidence_hits += 1

        results.append({
            "query": query,
            "expected": expected,
            "retrieved": retrieved_files,
            "hit": hit,
            "evidence": evidence,
            "valid": valid,
        })

    hit_rate = hits / len(SAMPLE_QUERIES)
    evidence_rate = evidence_hits / len(SAMPLE_QUERIES)
    return hit_rate, evidence_rate, results


# -----------------------------------------------------------
# Evaluation function
# -----------------------------------------------------------

def evaluate_retrieval(bot, top_k=3):
    """
    Runs DocuBot's retrieval system against SAMPLE_QUERIES.
    Returns a tuple: (hit_rate, detailed_results)

    hit_rate: fraction of queries where at least one retrieved snippet's
              filename matched an expected filename.
    detailed_results: list of dictionaries with structured info.
    """
    results = []
    hits = 0

    for query in SAMPLE_QUERIES:
        expected = expected_files_for_query(query)
        retrieved = bot.retrieve(query, top_k=top_k)

        retrieved_files = [fname for fname, _ in retrieved]

        hit = any(f in retrieved_files for f in expected) if expected else False
        if hit:
            hits += 1

        results.append({
            "query": query,
            "expected": expected,
            "retrieved": retrieved_files,
            "hit": hit
        })

    hit_rate = hits / len(SAMPLE_QUERIES)
    return hit_rate, results


# -----------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------

def print_eval_results(hit_rate, results):
    """
    Nicely formats evaluation results.
    """
    print("\nEvaluation Results")
    print("------------------")
    print(f"Hit rate: {hit_rate:.2f}\n")

    for item in results:
        print(f"Query: {item['query']}")
        print(f"  Expected:  {item['expected']}")
        print(f"  Retrieved: {item['retrieved']}")
        print(f"  Hit:       {item['hit']}")
        print()


# -----------------------------------------------------------
# Optional CLI entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    from docubot import DocuBot

    print("Running retrieval evaluation...\n")
    bot = DocuBot()

    hit_rate, results = evaluate_retrieval(bot)
    print_eval_results(hit_rate, results)
