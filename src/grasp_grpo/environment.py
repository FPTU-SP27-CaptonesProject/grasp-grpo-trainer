"""Project A-compatible RAG state transitions with bounded binary-tree expansion."""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import ROLE_MAX_TOKENS, candidate_index, parse, prompt
from .model import Action, generate
from .reader import ANSWER_RE, THOUGHT_RE  # noqa: F401  (re-exported for chat.py)
from .reader import legacy_reader_prompt as reader_prompt  # noqa: F401  (re-exported for chat.py)

# (role, prompt, samples) -> sampled proxy actions
GenerateFn = Callable[[str, str, int], list[Action]]


def _f1(prediction: str, gold: list[str]) -> float:
    import string
    from collections import Counter

    def tokens(text: str) -> list[str]:
        # Mirrors grasp-proxy-rollout.metrics.normalize_answer exactly.
        text = text.lower()
        text = "".join(char for char in text if char not in string.punctuation)
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        return text.split()
    p = tokens(prediction)
    scores = []
    for answer in gold:
        g = tokens(answer)
        common = sum((Counter(p) & Counter(g)).values())
        scores.append(0.0 if not common else 2 * common / (len(p) + len(g)))
    return max(scores, default=0.0)


def reward(answer: str, example: dict[str, Any], context: list[dict[str, Any]], retrieval_count: int, alpha: float, retrieval_cost: float, action_shaping: float = 0.0) -> dict[str, float | int | None]:
    f1 = _f1(answer, example["gold_answers"])
    gold_ids = set(example.get("gold_doc_ids", []))
    recall = len(gold_ids & {x["doc_id"] for x in context}) / len(gold_ids) if gold_ids else None
    total = f1 if recall is None else (1 - alpha) * f1 + alpha * recall
    sequence_reward = total - retrieval_cost * retrieval_count
    return {"answer_f1": f1, "context_recall": recall, "retrieval_count": retrieval_count, "retrieval_cost": retrieval_cost * retrieval_count, "sequence_reward": sequence_reward, "action_shaping": action_shaping, "total": sequence_reward + action_shaping}


@dataclass
class TreeNode:
    node_id: str
    parent_id: str | None
    depth: int
    question: str
    context: list[dict[str, Any]]
    thoughts: list[str]
    actions: list[dict[str, Any]] = field(default_factory=list)
    reader_response: str | None = None
    answer: str | None = None
    terminal: str | None = None
    children: list[str] = field(default_factory=list)
    transcript: str = ""


class BinaryTreeEnvironment:
    def __init__(self, actor, tokenizer, reader, retriever, *, branch_factor: int, max_steps: int, max_leaves: int, temperature: float, top_p: float, alpha: float, retrieval_cost: float, behavior_checkpoint: str, initial_retrieval: bool = True, rd_gain_weight: float = 0.3, qf_gain_weight: float = 0.7, ki_hit_weight: float = 0.3, ki_miss_penalty: float = 0.1, shaping_scale: float = 1.0, advantage_std_norm: bool = False, generate_fn: GenerateFn | None = None):
        if branch_factor != 2:
            raise ValueError("This implementation intentionally fixes binary-tree branch_factor to 2")
        self.actor, self.tokenizer, self.reader, self.retriever = actor, tokenizer, reader, retriever
        self.branch_factor, self.max_steps, self.max_leaves = branch_factor, max_steps, max_leaves
        self.temperature, self.top_p, self.alpha, self.retrieval_cost = temperature, top_p, alpha, retrieval_cost
        self.behavior_checkpoint, self.initial_retrieval = behavior_checkpoint, initial_retrieval
        self.rd_gain_weight, self.qf_gain_weight = rd_gain_weight, qf_gain_weight
        self.ki_hit_weight, self.ki_miss_penalty = ki_hit_weight, ki_miss_penalty
        self.shaping_scale, self.advantage_std_norm, self.generate_fn = shaping_scale, advantage_std_norm, generate_fn

    @staticmethod
    def _recall(gold_ids: set[str], passages: list[dict[str, Any]]) -> float | None:
        if not gold_ids:
            return None
        return len(gold_ids & {str(item["doc_id"]) for item in passages}) / len(gold_ids)

    @staticmethod
    def _shape(action: dict[str, Any], **values: float) -> None:
        action["reward_shaping"] = {key: float(value) for key, value in values.items()}
        action["reward_shaping"]["total"] = float(sum(values.values()))

    def _sample(self, role: str, state: dict[str, Any], samples: int) -> list[dict[str, Any]]:
        text_prompt = prompt(role, state)
        if self.generate_fn is not None:
            actions: list[Action] = self.generate_fn(role, text_prompt, samples)
        else:
            actions = generate(self.actor, self.tokenizer, text_prompt, samples=samples, max_new_tokens=ROLE_MAX_TOKENS[role], temperature=self.temperature, top_p=self.top_p)
        result = []
        for action in actions:
            parsed, valid, error = parse(role, action.text)
            result.append({"role": role, "prompt": text_prompt, "response": action.text, "parsed_action": parsed, "valid_format": valid, "validation_error": error, "response_token_ids": action.token_ids, "old_token_logprobs": action.token_logprobs, "old_logprob_sum": float(sum(action.token_logprobs)) if action.token_logprobs else None, "behavior_policy_checkpoint": self.behavior_checkpoint})
        return result

    def _leaf(self, node: TreeNode, reason: str) -> TreeNode:
        node.answer = self.reader.final(node.question, node.context, node.thoughts, node.transcript)
        node.terminal = reason
        return node

    def _next_nodes(self, node: TreeNode, node_number: list[int], gold_ids: set[str], question_baseline: list[dict[str, Any]]) -> list[TreeNode]:
        """One reader thought plus RD -> optional QF -> KI, branching only at RD.

        Keeping QF/KI to one sample per selected RD branch caps the binary tree while all
        proxy actions remain on-policy and receive the descendant leaf advantage.
        """
        step = self.reader.step(node.question, node.context, node.thoughts, node.transcript)
        if step.answer is not None:
            node.reader_response, node.answer, node.terminal = step.piece, step.answer, "reader_answer"
            node.transcript += step.piece
            return []
        if step.thought is None:
            self._leaf(node, "reader_empty")
            return []
        thought = step.thought
        state = {"question": node.question, "context": node.context, "thoughts": node.thoughts + [thought], "current_thought": thought}
        children = []
        for rd in self._sample("rd", state, self.branch_factor):
            child = copy.deepcopy(node)
            child.node_id, child.parent_id, child.depth = f"n{node_number[0]}", node.node_id, node.depth + 1
            node_number[0] += 1
            child.thoughts.append(thought)
            child.transcript = node.transcript + step.piece
            child.reader_response = step.piece
            child.actions = [rd]
            retrieve = rd["parsed_action"] if rd["valid_format"] else True
            if not rd["valid_format"]:
                rd["environment_fallback"] = "invalid RD defaults to retrieval"
            if retrieve:
                qf = self._sample("qf", state, 1)[0]
                child.actions.append(qf)
                parsed_qf = qf["parsed_action"] if qf["valid_format"] else None
                query = str(parsed_qf.get("query") or node.question) if parsed_qf else node.question
                passages = self.retriever.search(query)
                remaining_gold = gold_ids - {str(item["doc_id"]) for item in node.context}
                if remaining_gold:
                    baseline_recall = self._recall(remaining_gold, question_baseline) or 0.0
                    query_recall = self._recall(remaining_gold, passages) or 0.0
                    qf_gain = self.qf_gain_weight * max(0.0, query_recall - baseline_recall) ** 0.5
                else:
                    qf_gain = 0.0
                self._shape(qf, query_recall_gain=qf_gain)
                candidates = self.retriever.candidate_triples(query, passages, question=node.question)
                if candidates:
                    ki_state = {"question": node.question, "chain": parsed_qf.get("chain", []) if parsed_qf else [], "query": query, "candidates": candidates}
                    ki = self._sample("ki", ki_state, 1)[0]
                    child.actions.append(ki)
                    chosen = candidate_index(ki["parsed_action"] if ki["valid_format"] else None, candidates)
                    if chosen is None:
                        chosen, ki["environment_fallback"] = 0, "invalid KI selected candidate 0"
                    selected = candidates[chosen]
                    evidence = next((x for x in passages if x["doc_id"] == selected["doc_id"]), passages[0])
                    if ki["valid_format"] and str(evidence["doc_id"]) in gold_ids:
                        rank = next((index + 1 for index, item in enumerate(passages) if item["doc_id"] == evidence["doc_id"]), 1)
                        ki_reward = self.ki_hit_weight / math.log2(rank + 1)
                    elif gold_ids:
                        ki_reward = -self.ki_miss_penalty
                    else:
                        ki_reward = 0.0
                    self._shape(ki, evidence_selection=ki_reward)
                    if all(x["doc_id"] != evidence["doc_id"] for x in child.context):
                        child.context.append(evidence)
                else:
                    self._shape(qf, query_recall_gain=qf_gain)
            before = self._recall(gold_ids, node.context)
            after = self._recall(gold_ids, child.context)
            rd_gain = self.rd_gain_weight * ((after or 0.0) - (before or 0.0)) if before is not None else 0.0
            self._shape(rd, coverage_gain=rd_gain)
            node.children.append(child.node_id)
            children.append(child)
        return children

    def run(self, example: dict[str, Any]) -> dict[str, Any]:
        if self.actor is not None:  # None when a vLLM server generates
            self.actor.eval()
        select = getattr(self.retriever, "select", None)
        if select is not None:
            select(example)
        initial = self.retriever.search(example["question"]) if self.initial_retrieval else []
        question_baseline = list(initial)
        gold_ids = {str(value) for value in example.get("gold_doc_ids", [])}
        root = TreeNode("n0", None, 0, example["question"], initial, [])
        nodes: dict[str, TreeNode] = {root.node_id: root}
        frontier, leaves, counter = [root], [], [1]
        while frontier and len(leaves) < self.max_leaves:
            node = frontier.pop(0)
            if node.depth >= self.max_steps:
                leaves.append(self._leaf(node, "max_steps")); continue
            children = self._next_nodes(node, counter, gold_ids, question_baseline)
            if node.terminal:
                leaves.append(node); continue
            for child in children:
                nodes[child.node_id] = child
                if len(frontier) + len(leaves) < self.max_leaves:
                    frontier.append(child)
        for node in frontier:
            leaves.append(self._leaf(node, "leaf_cap"))
        records = []
        for leaf in leaves[: self.max_leaves]:
            chain, current = [], leaf
            while current:
                chain.append(current.node_id)
                current = nodes.get(current.parent_id) if current.parent_id else None
            chain.reverse()
            retrievals = sum(action["role"] == "rd" and bool(action.get("parsed_action")) for nid in chain for action in nodes[nid].actions)
            actions = [action for nid in chain for action in nodes[nid].actions]
            shaping = self.shaping_scale * sum(float(action.get("reward_shaping", {}).get("total", 0.0)) for action in actions)
            records.append({"leaf_id": leaf.node_id, "path": chain, "question_id": example["id"], "question": example["question"], "gold_answers": example["gold_answers"], "final_answer": leaf.answer, "terminal": leaf.terminal, "reward": reward(leaf.answer or "", example, leaf.context, retrievals, self.alpha, self.retrieval_cost, shaping), "actions": actions})
        values = [float(x["reward"]["total"]) for x in records]
        mean = sum(values) / len(values) if values else 0.0
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5 if values else 0.0
        for record in records:
            centered = float(record["reward"]["total"]) - mean
            if std < 1e-6:
                record["advantage"] = 0.0
            elif self.advantage_std_norm:
                record["advantage"] = centered / (std + 1e-6)
            else:
                record["advantage"] = centered  # Dr. GRPO: no std division
        return {"question_id": example["id"], "question": example["question"], "tree": {node_id: {"parent_id": node.parent_id, "depth": node.depth, "children": node.children, "terminal": node.terminal} for node_id, node in nodes.items()}, "leaves": records, "reward_mean": mean, "reward_std": std, "zero_signal": std < 1e-6, "behavior_policy_checkpoint": self.behavior_checkpoint}
