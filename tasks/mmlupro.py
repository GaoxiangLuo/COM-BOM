from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from datasets import load_dataset
from openai import OpenAI

from utils.evaluation import evaluate_correctness

from .base import BaseTask, MetricSpec

PROMPT_PREFIX = (
    "The following are multiple choice questions (with examples) about {task}. "
    "When you provide the answer to the last question, please use the option "
    "letter without any modification, and provide the answer directly, with no "
    "formatting, no bolding, and no markup. For example, (A). The final answer "
    "MUST only be the letter corresponding to the correct answer.\n"
)

OPTION_LETTERS = list("ABCDEFGHIJKLMNOP")


def compute_calibration_error(confidence, correct, p="2", beta=10):
    confidence = np.array(confidence)
    correct = np.array(correct)
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if len(confidence) % beta != 0:
        bins.append([len(confidence) - len(confidence) % beta, len(confidence)])
    elif bins:
        bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for start, end in bins:
        bin_confidence = confidence[start:end]
        bin_correct = correct[start:end]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in {"infty", "infinity", "max"}:
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = np.sqrt(cerr)

    return float(cerr)


class MMLUPro(BaseTask):
    """Multi-objective exemplar selection task on MMLU-Pro."""

    task_name = "mmlupro"
    _decision_dim = 32
    _metric_specs = (
        MetricSpec(name="accuracy", goal="maximize"),
        MetricSpec(name="calibration_error", goal="minimize"),
    )

    def __init__(
        self,
        *,
        task: str = "business",
        n: int = 8,
        model: str = "Qwen/Qwen3-8B",
        temperature: float = 0.7,
        port: int = 8000,
        num_workers: int = 32,
        text_embedding_model: str | None = None,
        prompt_type: str | None = None,
    ) -> None:
        self.task = task
        self.n = n
        self.model = model
        self.temperature = temperature
        self.num_workers = num_workers
        self.dtype = torch.float64
        self.text_embedding_model = text_embedding_model or "dunzhang/stella_en_400M_v5"
        self.prompt_type = prompt_type or "s2s_query"
        self._baseline_resources: dict[str, object] | None = None

        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        dataset = dataset.filter(lambda x: x["category"] == task.replace("_", " "))
        dataset_split = dataset.train_test_split(test_size=self._decision_dim + 1, seed=42)

        self._exemplar_dataset = dataset_split["test"]
        self._format_index = len(self._exemplar_dataset) - 1

        val_test_split = dataset_split["train"].train_test_split(test_size=0.5, seed=42)
        self.validation_data = val_test_split["test"]
        self.test_data = val_test_split["train"]

        if len(self.validation_data) == 0 or len(self.test_data) == 0:
            raise ValueError("Validation and test splits must be non-empty")

        self.client = OpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="local_deployment",
        )

    @property
    def decision_dim(self) -> int:
        return self._decision_dim

    @property
    def metrics(self) -> Sequence[MetricSpec]:
        return self._metric_specs

    def evaluate(self, X: torch.Tensor, *, split: str = "validation") -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != self.decision_dim:
            raise ValueError(
                f"Expected X with shape (batch, {self.decision_dim}), got {tuple(X.shape)}"
            )
        dataset = self._get_dataset(split)
        result = torch.zeros(X.shape[0], len(self.metrics), dtype=self.dtype)
        X_cpu = X.detach().cpu()

        for row_idx, candidate in enumerate(X_cpu):
            exemplar_prompt = self._build_exemplar_prompt(candidate)
            correct_list: list[float] = []
            confidence_list: list[float] = []

            def process_item(
                item: dict, *, prompt_template: str = exemplar_prompt
            ) -> tuple[float, float]:
                prompt = self._format_prompt(prompt_template, item)
                completion_kwargs = dict(
                    model=self.model,
                    prompt=prompt,
                    n=self.n,
                    temperature=self.temperature,
                    max_tokens=3,
                    stop=["Question:"],
                )
                completion = self.client.completions.create(**completion_kwargs)
                prediction, confidence = self._majority_vote(completion.choices)
                is_correct = evaluate_correctness(prediction, item["answer"])
                return is_correct, confidence

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_item, item): item for item in dataset}
                for future in as_completed(futures):
                    try:
                        correct, confidence = future.result()
                        correct_list.append(correct)
                        confidence_list.append(confidence)
                    except Exception as exc:  # pragma: no cover - logging side-effect
                        print(f"Generated an exception: {exc}")

            accuracy = sum(correct_list) / (len(correct_list) + 1e-8)
            calibration_error = compute_calibration_error(confidence_list, correct_list)
            result[row_idx, 0] = accuracy
            result[row_idx, 1] = calibration_error
            print(
                (
                    "Exemplars: {cand}, Number of exemplars: {n_ex}, "
                    "Accuracy: {acc:.4f}, Calibration Error: {cal_err:.4f}"
                ).format(
                    cand=candidate,
                    n_ex=int(candidate.sum().item()),
                    acc=accuracy,
                    cal_err=calibration_error,
                )
            )
        return result

    def _get_dataset(self, split: str) -> list[dict]:
        if split == "validation":
            return self.validation_data
        if split == "test":
            return self.test_data
        raise ValueError("split must be 'validation' or 'test'")

    def _build_exemplar_prompt(self, candidate: torch.Tensor) -> str:
        prompt_parts = [
            self._format_example_text(
                self._exemplar_dataset[self._format_index], include_answer=True
            )
        ]
        for idx, flag in enumerate(candidate.tolist()):
            if idx >= self._format_index:
                break
            if flag >= 0.5:
                prompt_parts.append(
                    self._format_example_text(self._exemplar_dataset[idx], include_answer=True)
                )
        return "".join(prompt_parts)

    def _format_prompt(self, exemplar_prompt: str, item: dict) -> str:
        prompt = PROMPT_PREFIX.format(task=self.task)
        prompt += exemplar_prompt
        prompt += self._format_example_text(item, include_answer=False)
        prompt += "The answer is: "
        return prompt

    def _format_example_text(self, example: dict, *, include_answer: bool) -> str:
        lines = ["Question:\n", f"{example['question']}\n", "Options:\n"]
        for idx, option in enumerate(example["options"]):
            letter = OPTION_LETTERS[idx]
            lines.append(f"{letter}. {option}\n")
        if include_answer:
            lines.append(f"The answer is: ({example['answer']})\n")
        return "".join(lines)

    def _baseline_anchor_prompt(self) -> str:
        anchor = self._exemplar_dataset[self._format_index]
        prompt = self._format_example_text(anchor, include_answer=False)
        prompt += f"The answer is: ({anchor['answer']})\n"
        return prompt

    def _baseline_prompt_for_candidate(self, candidate: torch.Tensor) -> str:
        base_prompt = self._baseline_anchor_prompt()
        exemplar_texts: list[str] = []
        for idx, flag in enumerate(candidate.tolist()):
            if idx >= self._format_index:
                break
            if flag >= 0.5:
                example = self._exemplar_dataset[idx]
                text = self._format_example_text(example, include_answer=False)
                text += f"The answer is: ({example['answer']})\n"
                exemplar_texts.append(text)
        return base_prompt + "".join(exemplar_texts)

    def _ensure_baseline_resources(self) -> dict[str, object]:
        if self._baseline_resources is not None:
            return self._baseline_resources
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SentenceTransformer is required for baseline evaluation.") from exc
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("faiss is required for baseline evaluation.") from exc

        exemplar_texts = [
            self._format_example_text(self._exemplar_dataset[idx], include_answer=False)
            for idx in range(self._format_index)
        ]
        model = SentenceTransformer(self.text_embedding_model, trust_remote_code=True)
        if torch.cuda.is_available():
            model = model.cuda()
        encode_kwargs = dict(normalize_embeddings=True, show_progress_bar=False)
        prompt_name = None
        if self.prompt_type:
            encode_kwargs["prompt_name"] = self.prompt_type
            prompt_name = self.prompt_type
        try:
            key_embeddings = model.encode(exemplar_texts, **encode_kwargs)
        except TypeError:
            encode_kwargs.pop("prompt_name", None)
            prompt_name = None
            key_embeddings = model.encode(exemplar_texts, **encode_kwargs)
        key_embeddings = np.asarray(key_embeddings, dtype=np.float32)
        faiss_index = faiss.IndexFlatIP(key_embeddings.shape[1])
        faiss_index.add(key_embeddings)
        self._baseline_resources = {
            "prompt_name": prompt_name,
            "model": model,
            "embeddings": key_embeddings,
            "index": faiss_index,
            "faiss": faiss,
        }
        return self._baseline_resources

    def _evaluate_rag_matrix(self, selections: torch.Tensor) -> tuple[float, float]:
        dataset = list(self.test_data)
        if len(selections) != len(dataset):
            raise ValueError("Selection matrix must align with test dataset length.")
        correct_list: list[float] = []
        confidence_list: list[float] = []
        for candidate, item in zip(selections, dataset):  # noqa: B905 - Python <3.10
            prompt_prefix = self._baseline_prompt_for_candidate(candidate.to(dtype=self.dtype))
            prompt = PROMPT_PREFIX.format(task=self.task)
            prompt += prompt_prefix
            prompt += self._format_example_text(item, include_answer=False)
            prompt += "The answer is: "
            completion = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                n=self.n,
                temperature=self.temperature,
                max_tokens=3,
                stop=["Question:"],
            )
            prediction, confidence = self._majority_vote(completion.choices)
            correct = evaluate_correctness(prediction, item["answer"])
            correct_list.append(float(correct))
            confidence_list.append(float(confidence))
        if not correct_list:
            raise RuntimeError("No baseline evaluations were completed.")
        accuracy = sum(correct_list) / (len(correct_list) + 1e-8)
        calibration_error = compute_calibration_error(confidence_list, correct_list)
        return accuracy, calibration_error

    def nearest_baseline_metrics(self, k: int) -> tuple[float, float]:
        resources = self._ensure_baseline_resources()
        dataset = list(self.test_data)
        if not dataset:
            raise RuntimeError("Test dataset is empty.")
        prompt_name = resources["prompt_name"]
        model = resources["model"]
        index = resources["index"]
        key_embeddings = resources["embeddings"]
        k_eff = max(0, min(k, key_embeddings.shape[0]))
        selections = torch.zeros(len(dataset), self.decision_dim, dtype=self.dtype)
        if k_eff > 0:
            query_texts = [
                self._format_example_text(item, include_answer=False) for item in dataset
            ]
            encode_kwargs = dict(normalize_embeddings=True, show_progress_bar=False)
            if prompt_name:
                encode_kwargs["prompt_name"] = prompt_name
            try:
                query_embeddings = model.encode(query_texts, **encode_kwargs)
            except TypeError:
                encode_kwargs.pop("prompt_name", None)
                query_embeddings = model.encode(query_texts, **encode_kwargs)
            query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
            _, indices = index.search(query_embeddings, k_eff)
            for row_idx, row in enumerate(indices):
                for idx_val in row:
                    if 0 <= idx_val < self.decision_dim:
                        selections[row_idx, idx_val] = 1.0
        accuracy, calibration_error = self._evaluate_rag_matrix(selections)
        return accuracy, calibration_error

    def diversity_baseline_metrics(self, k: int) -> tuple[float, float]:
        resources = self._ensure_baseline_resources()
        key_embeddings = resources["embeddings"]
        faiss_index = resources["index"]
        faiss = resources["faiss"]
        if key_embeddings.size == 0:
            raise RuntimeError("No exemplar embeddings available for diversity baseline.")
        k_eff = max(1, min(k, key_embeddings.shape[0]))
        kmeans = faiss.Kmeans(key_embeddings.shape[1], k_eff, niter=100, verbose=False)
        kmeans.train(key_embeddings)
        _, assignment = faiss_index.search(kmeans.centroids, 1)
        candidate = torch.zeros(1, self.decision_dim, dtype=self.dtype)
        for idx_val in assignment.flatten():
            if 0 <= idx_val < self.decision_dim:
                candidate[0, idx_val] = 1.0
        metrics = self.evaluate(candidate, split="test")
        return metrics[0, 0].item(), metrics[0, 1].item()

    def _majority_vote(self, choices: Iterable) -> tuple[str, float]:
        choice_list = list(choices)
        clusters: dict[str, list] = defaultdict(list)
        for choice in choice_list:
            prediction = choice.text.strip()
            matched = False
            for option in OPTION_LETTERS:
                if evaluate_correctness(prediction, option):
                    clusters[option].append(choice)
                    matched = True
                    break
            if not matched:
                clusters[prediction].append(choice)
        if not clusters:
            raise RuntimeError("Model returned no choices")
        largest = max(clusters.items(), key=lambda item: (len(item[1]), item[0]))
        confidence = len(largest[1]) / max(len(choice_list), 1)
        return largest[0], confidence


__all__ = ["MMLUPro"]
