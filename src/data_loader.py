from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from utils import setup_logger

logger = setup_logger(__name__)

@dataclass
class EvalQuestion:
    """Represents a standardized evaluation question with its gold answer(s)."""
    id: str
    question: str
    answers: List[str]
    context: Optional[str] = None
    type: str = "generic"

class DatasetLoader:
    """Base class for dataset loading and preprocessing."""

    def __init__(self, name: str):
        self.name = name
        self.dataset: Optional[Dataset] = None

    def load(self) -> Dataset:
        raise NotImplementedError

    def preprocess(self) -> List[str]:
        raise NotImplementedError

    def get_eval_questions(self) -> List[EvalQuestion]:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Calculates basic statistics for the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please call `load()` first.")
        
        texts = self.preprocess()
        n_docs = len(texts)
        tokens = [len(text.split()) for text in texts]
        n_tokens = sum(tokens)
        vocab_size = len(set(word for text in texts for word in text.split()))
        avg_doc_length = sum(tokens) / len(tokens) if len(tokens) > 0 else 0

        return {
            "n_docs": n_docs,
            "n_tokens": n_tokens,
            "vocab_size": vocab_size,
            "avg_doc_length": avg_doc_length,
        }

class MultiHopRAGLoader(DatasetLoader):
    """DatasetLoader for the MultiHop-RAG dataset."""

    def __init__(self):
        super().__init__("MultiHop-RAG")

    def load(self) -> Dataset:
        logger.info(f"Loading {self.name} dataset...")
        dataset = load_dataset("yixuantt/MultiHopRAG", split="train")
        self.dataset = dataset
        return self.dataset

    def preprocess(self) -> List[str]:
        if self.dataset is None:
            self.load()
        # The dataset contains context strings
        return [item["context"] for item in self.dataset if "context" in item]

    def get_eval_questions(self) -> List[EvalQuestion]:
        if self.dataset is None:
            self.load()
        questions = []
        for i, item in enumerate(self.dataset):
            questions.append(
                EvalQuestion(
                    id=f"mh_{i}",
                    question=item["question"],
                    answers=[item["answer"]],
                    context=item.get("context", ""),
                    type="multihop"
                )
            )
        return questions

class HotPotQALoader(DatasetLoader):
    """DatasetLoader for the HotPotQA dataset (distractor subset)."""

    def __init__(self):
        super().__init__("HotPotQA")

    def load(self) -> Dataset:
        logger.info(f"Loading {self.name} dataset...")
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        self.dataset = dataset
        return self.dataset

    def preprocess(self) -> List[str]:
        if self.dataset is None:
            self.load()
        texts = []
        for item in self.dataset:
            # Flatten context items into single strings
            for title, sentences in item["context"]:
                text = f"Title: {title}. " + " ".join(sentences)
                texts.append(text)
        return list(set(texts))

    def get_eval_questions(self) -> List[EvalQuestion]:
        if self.dataset is None:
            self.load()
        questions = []
        for item in self.dataset:
            context_str = " ".join([f"{t}: " + " ".join(s) for t, s in item["context"]])
            questions.append(
                EvalQuestion(
                    id=item["id"],
                    question=item["question"],
                    answers=[item["answer"]],
                    context=context_str,
                    type=item["type"]
                )
            )
        return questions

class SyntheticCorpusLoader(DatasetLoader):
    """DatasetLoader for the custom synthetic corpus."""

    def __init__(self):
        super().__init__("SyntheticCorpus")
        # In a real implementation we would load from a generated JSON/JSONL
        self.mock_data = [{"text": f"Synthetic domain document {i}", "question": "What is this?", "answer": "Synthetic document"} for i in range(500)]

    def load(self) -> Dataset:
        logger.info(f"Loading {self.name} dataset...")
        self.dataset = Dataset.from_list(self.mock_data)
        return self.dataset

    def preprocess(self) -> List[str]:
        if self.dataset is None:
            self.load()
        return [item["text"] for item in self.dataset]

    def get_eval_questions(self) -> List[EvalQuestion]:
        if self.dataset is None:
            self.load()
        questions = []
        for i, item in enumerate(self.dataset):
            questions.append(
                EvalQuestion(
                    id=f"syn_{i}",
                    question=item["question"],
                    answers=[item["answer"]],
                    context=item["text"],
                    type="local"
                )
            )
        return questions
