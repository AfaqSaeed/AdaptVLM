"""Connect model inference to immutable feedback snapshots."""

from PIL import Image

from adaptvlm.feedback.store import FeedbackStore, text
from adaptvlm.inference.engine import InferenceEngine


class FeedbackService:
    """Generate candidates with provenance and save exactly what users review."""

    def __init__(
        self,
        store: FeedbackStore,
        engine: InferenceEngine,
        comparison: InferenceEngine | None = None,
    ):
        self.store, self.engine, self.comparison = store, engine, comparison

    @staticmethod
    def candidate(engine: InferenceEngine, image: Image.Image, question: str) -> dict:
        """Generate a successful candidate, including resolved checkpoint and settings."""
        prediction = engine.predict(image, question)
        if prediction.error:
            raise ValueError(prediction.error)
        return prediction.to_dict() | {
            "model_id": engine.config.model_id,
            "resolved_revision": getattr(engine.backend, "resolved_revision", None),
            "generation_parameters": engine.config.to_dict() | {"do_sample": False},
            "adapter_files": (getattr(engine.backend, "adapter_metadata", None) or {}).get(
                "adapter_files"
            ),
        }

    def generate(
        self,
        image: Image.Image,
        question: str,
        split: str = "unassigned",
        pair: bool = False,
        session_id: str | None = None,
    ) -> dict:
        """Store a single answer or a blinded response pair after successful generation."""
        question = text(question, "Question", 4000)
        if pair and self.comparison is None:
            raise ValueError("Pair comparisons require --comparison-config for a second candidate")
        candidates = [self.candidate(self.engine, image, question)]
        if pair:
            candidates.append(self.candidate(self.comparison, image, question))
        return self.store.create_request(image, question, candidates, split, session_id)
