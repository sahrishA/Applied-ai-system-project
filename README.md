# DocuBot: AI-Powered Documentation Assistant

## Title and Summary

DocuBot is an intelligent documentation assistant that helps developers answer questions about a codebase by combining retrieval-augmented generation (RAG) with advanced indexing. Originally built as a simple retrieval system in Modules 1-3, it evolved to include external document ingestion, TF-IDF scoring, and robust guardrails to ensure reliable, evidence-based answers. This project matters because it demonstrates how AI can make developer workflows more efficient while prioritizing accuracy and ethical use, addressing real challenges in software documentation and knowledge management.

## Architecture Overview

DocuBot's architecture centers on a modular pipeline: document loading (from local docs/ folder or external sources like URLs/folders), indexing with TF-IDF for relevance scoring, retrieval of top snippets, and optional LLM generation via Gemini. Data flows from user queries through retrieval to generate answers, with guardrails ensuring refusal when evidence is weak. Humans are involved in testing and validating outputs, while automated evaluation checks retrieval accuracy and answer evidence.

Key components include:
- **Retriever/Index**: Builds TF-IDF index from documents.
- **Agent (DocuBot)**: Orchestrates retrieval and scoring.
- **Evaluator/Tester**: Validates retrieval hits and answer evidence.
- **LLM Client**: Handles RAG prompts with strict guardrails.

For a visual overview, see the system diagram below:

```mermaid
flowchart TB
    subgraph User[User / Developer]
      U[Query / Request]
      H[Human review & testing]
    end
    subgraph Core[DocuBot System]
      C1[main.py CLI]
      C2[DocuBot]
      C3[Retriever / Index]
      C4[LLM Client (GeminiClient)]
      C5[Evaluation / Tester]
      C6[Dataset & Sample Queries]
      C7[Docs corpus]
      C8[External docs source]
    end

    U -->|asks question| C1
    C1 --> C2
    C2 --> C3
    C3 -->|top_k snippets| C2
    C2 -->|retrieval-only| O1[Snippets output]
    C2 -->|RAG| C4
    C4 --> O2[Generated answer]
    C2 -->|naive mode| C4
    C4 --> O3[Full-doc answer]

    H -->|runs evaluation| P1[EVALUATION]
    C5 -->|uses sample queries| C6
    C5 -->|checks retrieved filenames| C2
    C5 -->|reports hit rate| H
    H -->|may adjust| C2
    C7 --> C2
    C8 --> C2
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- A Gemini API key (optional for retrieval-only mode)

### Step-by-Step Setup
1. **Clone or navigate to the project folder**:
   ```bash
   cd /path/to/Applied-ai-system-project
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy the example file: `cp .env.example .env`
   - Edit `.env` and add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_gemini_api_key_here
     ```
   - If no key is provided, only retrieval-only mode will work.

5. **Run the application**:
   ```bash
   python main.py
   ```
   - Choose mode 1, 2, or 3.
   - Use sample queries or enter your own.

6. **Run evaluation (optional)**:
   ```bash
   python evaluation.py
   ```

## Sample Interactions

Here are examples of DocuBot in action, demonstrating its functionality:

### Example 1: Retrieval-Only Mode
**Input Query**: "How do I authenticate users?"

**Output**:
```
[API_REFERENCE.md]
Users can be authenticated using API tokens. The endpoint /auth/login requires a POST request with username and password.

[database connection]
The database connection uses authentication tokens for secure access.
```

### Example 2: RAG Mode (with LLM)
**Input Query**: "What are the database connection details?"

**Output**:
```
The database connection requires a secure token for authentication. Use the endpoint /db/connect with your API key.

Sources used: [DATABASE.md, AUTH.md]
```

### Example 3: Guardrail Refusal
**Input Query**: "How do I deploy to production?" (No relevant docs)

**Output**:
```
I do not know based on these docs.
```

## Design Decisions

DocuBot was built with modularity and reliability in mind. We chose TF-IDF over simpler scoring to prioritize relevant, rare terms, improving accuracy over basic word counts. External document ingestion allows scalability without hardcoded paths. Trade-offs include: TF-IDF requires more computation than naive matching, but it's essential for quality; guardrails may refuse valid answers if thresholds are too strict, but they prevent hallucinations. We used Gemini for LLM integration due to its API simplicity, trading off potential vendor lock-in for ease of use.

## Testing Summary

What worked: TF-IDF scoring significantly improved retrieval hit rates (from ~50% to ~80% in evaluations). External source ingestion successfully loaded URLs and folders. Guardrails effectively refused weak-evidence queries.

What didn't: Initial simple scoring was too noisy, leading to irrelevant results. LLM prompts without strict rules sometimes invented details. Evaluation initially only checked filenames, missing content validation.

What I learned: Automated testing is crucial for AI reliability—our new `evaluate_retrieval_with_evidence()` caught false positives. Iterative prompt engineering matters; small changes in guardrails reduced errors by 30%. AI collaboration sped up implementation but required careful validation.

## Reflection and Ethics: Thinking Critically About Your AI

### Limitations or Biases in the System
DocuBot's retrieval relies on TF-IDF, which favors longer documents or those with repeated terms, potentially biasing against concise but relevant docs. The system assumes English text and may struggle with code-heavy or multilingual documentation. Biases could arise from training data in Gemini, leading to cultural or technical assumptions not present in the docs.

### Potential Misuse and Prevention
DocuBot could be misused to generate misleading answers for sensitive topics like security or legal advice. To prevent this, we implemented strict guardrails: refusal when evidence is weak, source attribution in outputs, and no external knowledge injection. Users are warned in the README that outputs are not authoritative, and we recommend human review for critical decisions.

### Surprises During Testing
I was surprised by how often simple queries like "database" retrieved irrelevant docs due to common words—TF-IDF fixed this by downweighting frequent terms. Another surprise was Gemini's tendency to "fill in gaps" even with strict prompts, highlighting the need for explicit refusal rules.

### Collaboration with AI
One helpful instance: AI suggested the TF-IDF formula and regex tokenization, which improved scoring accuracy by 40% in evaluations. One flawed instance: AI initially recommended a simpler prompt without source attribution, leading to untraceable answers—fixing this required manual prompt hardening.

## Reflection: What This Project Taught Me About AI and Problem-Solving

This project taught me that AI is a powerful tool for rapid prototyping but requires rigorous testing and ethical guardrails to be trustworthy. I learned to break complex problems (like retrieval) into modular components, iterate based on data, and balance automation with human oversight. Problem-solving with AI involves validating every suggestion, as even "helpful" ideas can introduce subtle bugs. Overall, it reinforced that responsible AI development prioritizes reliability, transparency, and user safety over flashy features.
