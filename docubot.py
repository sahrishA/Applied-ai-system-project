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
        """
        """
            index = {}  # Plain dict
            for filename, text in documents:
                words = text.lower().split()  # Basic split on whitespace (keeps punctuation)
                for word in words:
                    if word not in index:
                        index[word] = []  # List of filenames
                    if filename not in index[word]:
                        index[word].append(filename)  # Just tracks presence (1 if present, 0 otherwise)
            return index
    Data structure: {word: [filename1, filename2, ...]}
        """
        index = defaultdict(dict) //dictionary where each word maps to another dictionary of filename to term frequency
        for filename, text in documents:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts = Counter(words)
            for word, count in word_counts.items():
                index[word][filename] = count
        return index
    """data structure: {word: {filename: count, ...}} (nested dict with term frequencies).
    What it stores: Frequency (how many times the word appears in each doc), enabling TF(term frequency) calculations.
    score_document (e.g., self.index[word][filename] directly gets the count).
    Formula: TF = (number of times the word appears in the document) / (total words in the document)"""
    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, filename):
        """
        Return a simple relevance score for how well the text matches the query.
        
        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        # origianal simple word counting approach:
        """
        query_words = query.lower().split()
        score = 0
        for word in query_words:
            if word in text.lower():
                score += 1
        return score
        """
        """
        Return a TF-IDF based relevance score for how well the query matches the document.

        Uses term frequency (TF) normalized by document length, and inverse document frequency (IDF).
        """
        """
        Term Frequency (TF):

    Measures how often a query word appears within a specific document.
    Formula: TF = (number of times the word appears in the document) / (total words in the document)
    Why normalize by document length? Longer documents naturally have more word occurrences, so this prevents them from unfairly scoring higher. For example:
    In a short doc (50 words) with "database" appearing 2 times: TF = 2/50 = 0.04
    In a long doc (500 words) with "database" appearing 2 times: TF = 2/500 = 0.004 (lower score, fairer)
    Inverse Document Frequency (IDF):

    Measures how rare or unique a word is across the entire corpus (all documents).
    Formula: IDF = log(total number of documents / number of documents containing the word)
    Why inverse? Common words (e.g., "the", "and") appear in many docs, so their IDF is low (close to 0). Rare words (e.g., "authentication") have high IDF, making them more valuable for distinguishing relevant docs.
    Example: If "database" appears in 3 out of 5 docs, IDF = log(5/3) ≈ log(1.67) ≈ 0.51. If "token" appears in 1 out of 5, IDF = log(5/1) ≈ log(5) ≈ 1.61 (higher weight).
    TF-IDF Score:

    Combines them: TF-IDF = TF * IDF
    For each query word, calculate TF-IDF for the document, then sum across all query words.
    Higher scores mean the document is more relevant (frequent use of rare query terms).
    In DocuBot's Code
    This runs for each unique word in the query, summing the TF-IDF contributions.
    Result: Documents are ranked by relevance, not just raw matches.
    Why It's Better Than Simple Word Counting
    Old approach (original DocuBot): Count how many query words appear anywhere in the text. A doc with 10 "the" matches scores the same as one with 10 "database" matches—ignores rarity and length.
    TF-IDF advantage: Prioritizes docs with concentrated, meaningful terms. For query "database connection":
    A doc mentioning "database" 5 times (rare term) scores higher than one mentioning "the" 5 times (common term).
    Prevents long, irrelevant docs from dominating results.
    Real-world impact: Powers search engines like Google. In DocuBot, it improves evaluation.py hit rates by surfacing better snippets for RAG/LLM modes.
    If a query word doesn't exist in the index, its contribution is 0. For very small corpora (few docs), IDF can be extreme—consider smoothing if needed. Let me know if you'd like code examples or tweaks!"""
        
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
