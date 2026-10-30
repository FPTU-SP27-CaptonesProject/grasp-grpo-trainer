"""The frozen prompt and parser contract learned by the Project A SFT adapter."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

ROLES = ("rd", "qf", "ki")
ROLE_MAX_TOKENS = {"rd": 16, "qf": 256, "ki": 128}
TRIPLE_RE = re.compile(r"<\s*([^;<>]+?)\s*;\s*([^;<>]+?)\s*;\s*([^<>]+?)\s*>")
QF_RE = re.compile(r"^\s*Reasoning\s+Chain\s*:\s*(.*?)\s*\n\s*Knowledge\s+Gap\s*:\s*(.*?)\s*\n\s*Query\s*:\s*(.*?)\s*$", re.I | re.S)

RD = '''You are a retrieval decision assistant. A reasoning model is answering a question step by step. Decide whether its latest reasoning step needs external information that must be retrieved before reasoning can safely continue.

Answer "Yes" when the current step needs a specific fact (entity, date, number, location, title, relation) that is not already stated, even in paraphrase, in the Retrieved Context or the Previous Reasoning; when it introduces a new entity whose facts are not yet supported; or when it expresses uncertainty ("I think", "probably", "not sure", "unknown") about knowledge that is missing from the context.

Answer "No" when the needed information is already in the Retrieved Context or can be directly inferred from it; when the step only restates the question, summarises, or draws a conclusion from facts already present; or when it relies only on general knowledge, definitions, or simple arithmetic. Uncertain wording alone is not a reason to retrieve if the fact is already in the context.

Output exactly one of the following and nothing else:
Yes (retrieval is required)
No (retrieval is not required)'''

QF = '''You are a query formulation assistant. A reasoning model is answering a question step by step. Help it close the gap between its current reasoning and the final answer by writing one focused retrieval query. Work in three steps.

Step 1 - Reasoning Chain: from the thoughts, extract only the facts that directly help answer the question, as knowledge triples <head; relation; tail>. Use only facts explicitly stated in the thoughts; skip speculative, uncertain or irrelevant content. Write "None" if there are no such facts.
Step 2 - Knowledge Gap: state the single next piece of information needed to move the reasoning toward the answer.
Step 3 - Query: write a concise search query for that information, containing the key entities and the relation. If the reasoning is already sufficient to answer the question, write "None".

Output exactly three lines in this format:
Reasoning Chain: <head1; relation1; tail1>, <head2; relation2; tail2>
Knowledge Gap: To continue answering the question, we need to know: [the missing information]
Query: [the search query, or None]'''

KI = '''You are a knowledge integration assistant. A reasoning model is answering a question step by step. From the candidate knowledge triples, select the single triple that is most useful for continuing the reasoning toward the answer.

The selected triple should logically extend the current reasoning chain and help bridge the gap to the final answer. Use the query as a soft hint about the current information need, but it may be noisy; prefer coherence with the reasoning chain and relevance to the question. Ignore unrelated, redundant or off-topic candidates.

Output only the selected triple, copied exactly from the candidate list in the form <head; relation; tail>, and nothing else.'''


def format_triple(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return "<{head}; {relation}; {tail}>".format(**{key: value.get(key, "") for key in ("head", "relation", "tail")})


def _context(items: list[dict[str, Any]] | None) -> str:
    return "\n\n".join(f"Title: {x.get('title') or x.get('doc_id') or 'Untitled'}\nText: {x.get('text', '')}" for x in (items or [])) or "None"


def prompt(role: str, state: dict[str, Any]) -> str:
    question = str(state["question"]).strip()
    thoughts = [str(x) for x in state.get("thoughts", []) if str(x).strip()]
    if role == "rd":
        current = str(state.get("current_thought") or thoughts[-1]).strip()
        previous = "\n".join(thoughts[:-1]) or "None"
        return f"{RD}\n\n---\nRetrieved Context:\n{_context(state.get('context'))}\n\nQuestion:\n{question}\n\nPrevious Reasoning:\n{previous}\n\nCurrent Reasoning Step:\n{current}\n\nOutput:\n"
    if role == "qf":
        thought_text = "\n".join(thoughts) or "None"
        return f"{QF}\n\n---\nQuestion: {question}\n\nThoughts:\n{thought_text}\n"
    candidates = state.get("candidates", [])
    candidate_text = "\n".join(f"[{i}] {format_triple(x)}" for i, x in enumerate(candidates, 1))
    chain = ", ".join(format_triple(x) for x in state.get("chain", [])) or "None"
    return f"{KI}\n\n---\nCandidate Triples:\n{candidate_text}\n\nQuestion: {question}\n\nReasoning Chain: {chain}\n\nQuery: {state.get('query') or 'None'}\n\nSelected Triple:\n"


def parse_triples(text: str) -> list[dict[str, str]]:
    return [{"head": m.group(1).strip(), "relation": m.group(2).strip(), "tail": m.group(3).strip()} for m in TRIPLE_RE.finditer(text)]


def parse(role: str, text: str) -> tuple[Any, bool, str | None]:
    cleaned = text.strip()
    if role == "rd":
        match = re.fullmatch(r"(Yes|No)(?:\s*\([^\r\n()]+\))?\s*[.!]?", cleaned, re.I)
        return ((match.group(1).casefold() == "yes") if match else None, bool(match), None if match else "invalid RD")
    if role == "qf":
        match = QF_RE.fullmatch(cleaned)
        if not match:
            return None, False, "invalid QF"
        chain_text, gap, query = (x.strip() for x in match.groups())
        chain = [] if chain_text.casefold() == "none" else parse_triples(chain_text)
        if chain_text.casefold() != "none" and not chain:
            return None, False, "QF chain has no triples"
        return {"chain": chain, "knowledge_gap": gap, "query": None if query.casefold() == "none" else query}, bool(gap), None if gap else "empty gap"
    triples = parse_triples(cleaned)
    valid = len(triples) == 1 and TRIPLE_RE.fullmatch(cleaned) is not None
    return (triples[0] if valid else None), valid, None if valid else "invalid KI"


def candidate_index(selected: dict[str, str] | None, candidates: list[dict[str, Any]]) -> int | None:
    if selected is None:
        return None
    normal = lambda value: " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())
    target = tuple(normal(selected.get(k)) for k in ("head", "relation", "tail"))
    for i, item in enumerate(candidates):
        if target == tuple(normal(item.get(k)) for k in ("head", "relation", "tail")):
            return i
    return None
