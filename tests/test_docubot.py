import os
import tempfile
import unittest
from unittest.mock import patch

from docubot import DocuBot
from llm_client import GeminiClient


class DummyModel:
    def __init__(self):
        self.prompt = None

    def generate_content(self, prompt):
        self.prompt = prompt
        class Response:
            pass

        response = Response()
        response.text = "This answer is based on the snippets.\nSources used: [AUTH.md]"
        return response


class TestDocuBot(unittest.TestCase):
    def test_load_documents_with_external_folder(self):
        with tempfile.TemporaryDirectory() as root:
            docs_dir = os.path.join(root, "docs")
            extra_dir = os.path.join(root, "external")
            os.makedirs(docs_dir)
            os.makedirs(extra_dir)

            with open(os.path.join(docs_dir, "main.md"), "w", encoding="utf8") as f:
                f.write("This is main documentation about database and auth.")

            with open(os.path.join(extra_dir, "external.md"), "w", encoding="utf8") as f:
                f.write("This is external documentation about tokens.")

            bot = DocuBot(docs_folder=docs_dir, external_sources=[extra_dir])
            loaded_names = {filename for filename, _ in bot.documents}

        self.assertEqual(loaded_names, {"main.md", "external.md"})

    def test_score_document_tfidf_ranks_relevant_doc(self):
        with tempfile.TemporaryDirectory() as root:
            docs_dir = os.path.join(root, "docs")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "relevant.md"), "w", encoding="utf8") as f:
                f.write("database connection database authentication")

            with open(os.path.join(docs_dir, "irrelevant.md"), "w", encoding="utf8") as f:
                f.write("unrelated text about cooking and travel")

            bot = DocuBot(docs_folder=docs_dir)

            relevant_score = bot.score_document("database authentication", "relevant.md")
            irrelevant_score = bot.score_document("database authentication", "irrelevant.md")

        self.assertGreater(relevant_score, irrelevant_score)

    def test_retrieve_guardrail_refuses_when_no_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            docs_dir = os.path.join(root, "docs")
            os.makedirs(docs_dir)

            with open(os.path.join(docs_dir, "doc.md"), "w", encoding="utf8") as f:
                f.write("This document covers authentication and tokens.")

            bot = DocuBot(docs_folder=docs_dir)
            results = bot.retrieve("totally unrelated query phrase")

        self.assertEqual(results, [])

    def test_answer_from_snippets_includes_sources_and_refusal(self):
        os.environ["GEMINI_API_KEY"] = "test"
        dummy_model = DummyModel()

        with patch("llm_client.genai.GenerativeModel", return_value=dummy_model):
            client = GeminiClient()
            answer = client.answer_from_snippets(
                "How do I authenticate?",
                [("AUTH.md", "This file describes auth token usage.")],
            )

        self.assertIn("Sources used:", answer)
        self.assertIn("[AUTH.md]", answer)
        self.assertEqual(
            client.answer_from_snippets("Anything", []),
            "I do not know based on the docs I have.",
        )


if __name__ == "__main__":
    unittest.main()
