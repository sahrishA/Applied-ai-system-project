"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder and external sources (folders or URLs)
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import urllib.request
import re
import math
from collections import defaultdict, Counter

class DocuBot:
    MIN_USEFUL_SCORE = 1
    def __init__(self, docs_folder="docs", external_sources=None, llm_client=None):
        """
        docs_folder: directory containing project documentation files
        external_sources: optional list of additional sources (folder paths or URLs)
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.external_sources = external_sources or []
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)
        
        # Precompute document stats for scoring
        self.total_docs = len(self.documents)
        self.doc_lengths = {}
        for filename, text in self.documents:
            words = re.findall(r'\b\w+\b', text.lower())
            self.doc_lengths[filename] = len(words)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files from docs_folder and external_sources.
        external_sources can include folder paths or URLs.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        
        # Load from main docs_folder
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith((".md", ".txt")):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        
        # Load from external sources
        for source in self.external_sources:
            if source.startswith(("http://", "https://")):
                # Handle URL
                try:
                    with urllib.request.urlopen(source) as response:
                        text = response.read().decode('utf-8')
                    # Extract filename from URL
                    filename = source.split('/')[-1] or f"external_doc_{len(docs)}"
                    if not filename.endswith((".md", ".txt")):
                        filename += ".txt"  # Assume plain text if no extension
                    docs.append((filename, text))
                except Exception as e:
                    print(f"Warning: Failed to fetch {source}: {e}")
            else:
                # Assume it's a folder path
                if os.path.isdir(source):
                    pattern = os.path.join(source, "*.*")
                    for path in glob.glob(pattern):
                        if path.endswith((".md", ".txt")):
                            with open(path, "r", encoding="utf8") as f:
                                text = f.read()
                            filename = os.path.basename(path)
                            docs.append((filename, text))
                else:
                    print(f"Warning: External source '{source}' is not a valid directory.")
        
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build an inverted index mapping words to documents with term frequencies.

        Improved tokenization: uses regex to extract words, ignores punctuation.
        Structure: {word: {filename: term_frequency}}

        Previous implementation (kept for reference):
        - indexed word presence only
        - used whitespace splitting instead of regex
        - could not calculate TF values

        Old version:
            index = {}
            for filename, text in documents:
                words = text.lower().split()
                for word in words:
                    if word not in index:
                        index[word] = []
                    if filename not in index[word]:
                        index[word].append(filename)
            return index
        """
        index = defaultdict(dict)
        for filename, text in documents:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            for word, count in word_counts.items():
                index[word][filename] = count
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, filename):
        """
        Return a TF-IDF based relevance score for how well the query matches the document.

        Uses term frequency (TF) normalized by document length, and inverse document frequency (IDF).

        Previous implementation (kept for reference):
        - counted the number of query words present in the document
        - used simple substring matching against raw text
        - did not normalize by document length or weigh rare terms

        Old version:
            query_words = query.lower().split()
            score = 0
            for word in query_words:
                if word in text.lower():
                    score += 1
            return score
        """
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        score = 0.0
        for word in query_words:
            if word in self.index and filename in self.index[word]:
                tf = self.index[word][filename] / self.doc_lengths[filename]
                df = len(self.index[word])
                idf = math.log(self.total_docs / df) if df > 0 else 0
                score += tf * idf
        return score

    def _is_useful_score(self, score):
        """Return whether a score is strong enough to provide useful context."""
        return score >= self.MIN_USEFUL_SCORE

    def retrieve(self, query, top_k=3):
        """
        Use the index and TF-IDF scoring to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        results = []
        for filename, text in self.documents:
            score = self.score_document(query, filename)
            results.append((score, filename, text))

        results.sort(reverse=True)

        # Guardrail: only keep documents with a minimum useful score.
        # If no document scores at least MIN_USEFUL_SCORE, we have no useful
        # context and should refuse rather than guess.
        useful_results = [
            (filename, text)
            for score, filename, text in results
            if self._is_useful_score(score)
        ]

        return useful_results[:top_k]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
