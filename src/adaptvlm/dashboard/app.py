"""Local image QA, human feedback, review and export interface."""

import shutil
import uuid

from adaptvlm.feedback.service import FeedbackService
from adaptvlm.feedback.store import FeedbackStore


def build_app(engine, store: FeedbackStore | None = None, comparison=None):
    """Construct a local UI with per-browser state and immutable review snapshots."""
    import gradio as gr

    store = store or FeedbackStore("data/feedback.sqlite3")
    service = FeedbackService(store, engine, comparison)

    def generate(image, question, split, pair, session):
        session = session or str(uuid.uuid4())
        try:
            request = service.generate(image, question, split, pair, session)
            answers = request["displayed_answers"]
            return (
                request["id"],
                image,
                request["question"],
                answers[0],
                answers[1] if pair else "",
                "Ready for feedback.",
                session,
            )
        except (ValueError, OSError, RuntimeError) as exc:
            return None, None, "", "", "", str(exc), session

    def correction(request_id, rating, replacement):
        try:
            if rating not in {"Correct", "Incorrect"}:
                raise ValueError("Select correct or incorrect")
            store.add_correction(request_id, rating == "Correct", replacement)
            return "Feedback saved. Generate a new answer for another review."
        except ValueError as exc:
            return str(exc)

    def preference(request_id, choice):
        try:
            store.add_preference(request_id, choice)
            return "Preference saved. Generate a new pair for another review."
        except ValueError as exc:
            return str(exc)

    def refresh():
        rows = store.records()
        return store.summary(), [
            {
                "feedback_id": r["feedback_id"],
                "question": r["question"],
                "answers": [r["candidates"][i]["answer"] for i in r["display_order"]],
                "kind": r["kind"],
                "correct": r["correct"],
                "corrected_answer": r["corrected_answer"],
                "preference": r["preference"],
                "split": r["split"],
                "protected": r["protected"],
                "training_eligible": r["training_eligible"],
                "created_at": r["feedback_at"],
            }
            for r in rows[-100:]
        ]

    def export(eligible_only):
        try:
            destination = store.database.parent / "exports" / str(uuid.uuid4())
            info = store.export(destination, eligible_only)
            archive = shutil.make_archive(str(destination), "zip", root_dir=destination)
            return archive, f"Exported and validated {info['count']} records."
        except (ValueError, OSError) as exc:
            return None, str(exc)

    with gr.Blocks(title="AdaptVLM feedback", analytics_enabled=False) as app:
        session = gr.State(None)
        gr.Markdown("# AdaptVLM Â· Feedback collection\nImages and reviews are saved locally.")
        if engine.config.backend == "mock":
            gr.Markdown("**MOCK MODE â€” software testing only. Excluded from training exports.**")
        gr.Markdown(f"Primary model: `{engine.config.model_id}`")
        for pair, title in ((False, "Answer and correct"), (True, "Compare responses")):
            with gr.Tab(title):
                request_id = gr.State(None)
                if pair:
                    gr.Markdown("Response order is randomized; identities are hidden while voting.")
                    if comparison is None:
                        gr.Markdown("Restart with `--comparison-config` to enable pair generation.")
                image = gr.Image(type="pil", label="Upload image")
                question = gr.Textbox(label="Question")
                split = gr.Dropdown(
                    ["unassigned", "train", "validation", "test"],
                    value="unassigned",
                    label="Data assignment",
                )
                gr.Markdown(
                    "Choose train only for adaptation data. Registered validation/test "
                    "images remain protected regardless of this selection."
                )
                button = gr.Button(
                    "Generate pair" if pair else "Generate answer",
                    interactive=not pair or comparison is not None,
                )
                saved_image = gr.Image(type="pil", label="Image being reviewed", interactive=False)
                saved_question = gr.Textbox(label="Question being reviewed", interactive=False)
                with gr.Row():
                    answer_a = gr.Textbox(
                        label="Response A" if pair else "Saved answer", interactive=False
                    )
                    answer_b = gr.Textbox(label="Response B", interactive=False, visible=pair)
                status = gr.Textbox(label="Status", interactive=False)
                button.click(
                    generate,
                    [image, question, split, gr.State(pair), session],
                    [request_id, saved_image, saved_question, answer_a, answer_b, status, session],
                    concurrency_limit=1,
                    concurrency_id="generation",
                    api_name="compare" if pair else "answer",
                )
                if pair:
                    choice = gr.Radio(
                        [
                            ("Response A is better", "a"),
                            ("Response B is better", "b"),
                            ("Both are acceptable", "both"),
                            ("Neither is acceptable", "neither"),
                        ],
                        label="Preference",
                    )
                    gr.Button("Save preference").click(
                        preference, [request_id, choice], status, api_name="prefer"
                    )
                    review_fields = [choice]
                else:
                    rating = gr.Radio(["Correct", "Incorrect"], label="Is the answer correct?")
                    replacement = gr.Textbox(label="Corrected answer (optional, if incorrect)")
                    gr.Button("Save correction feedback").click(
                        correction, [request_id, rating, replacement], status, api_name="correct"
                    )
                    review_fields = [rating, replacement]

                # Clear stale review state when input changes; feedback uses only saved IDs.
                def clear_single():
                    return None, None, "", "", "", "Generate a new answer.", None, ""

                def clear_pair():
                    return None, None, "", "", "", "Generate a new pair.", None

                for component in (image, question, split):
                    component.change(
                        clear_pair if pair else clear_single,
                        outputs=[
                            request_id,
                            saved_image,
                            saved_question,
                            answer_a,
                            answer_b,
                            status,
                            *review_fields,
                        ],
                        api_name=False,
                    )
        with gr.Tab("Review and export"):
            gr.Markdown(
                "Latest 100 records. Audit exports contain all feedback; **do not train "
                "on audit exports**. Eligible exports exclude unassigned, held-out, "
                "mock/synthetic, tie, and unusable feedback."
            )
            refresh_button = gr.Button("Refresh feedback")
            summary = gr.JSON(label="Collection summary")
            recent = gr.JSON(label="Recent feedback")
            refresh_button.click(refresh, outputs=[summary, recent], api_name="review")
            eligible = gr.Checkbox(value=True, label="Export training-eligible records only")
            export_button = gr.Button("Export validated bundle")
            download = gr.File(label="Download ZIP (JSONL, manifest and images)")
            export_status = gr.Textbox(label="Export status", interactive=False)
            export_button.click(export, eligible, [download, export_status], api_name="export")
    return app
