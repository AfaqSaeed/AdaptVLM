"""Assistant-only multimodal labels without truncating image-token sequences."""

from PIL import Image, ImageOps


class SFTCollator:
    """Render the model chat template and verify the token-level assistant boundary."""

    def __init__(
        self, processor, max_sequence_length: int, image_token_id: int, prompt_suffix: str = ""
    ):
        self.processor = processor
        self.limit = max_sequence_length
        self.image_token_id = image_token_id
        self.prompt_suffix = prompt_suffix
        processor.tokenizer.padding_side = "right"

    def __call__(self, examples: list[dict]) -> dict:
        """Return image tensors, masks, input IDs and assistant-only labels."""
        import torch

        images, prompts, conversations = [], [], []
        for row in examples:
            with Image.open(row["image_path"]) as image:
                images.append([ImageOps.exif_transpose(image).convert("RGB")])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["question"] + self.prompt_suffix},
                    ],
                }
            ]
            prompts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]}
            )
            conversations.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
        settings = {"images": images, "return_tensors": "pt", "padding": True, "truncation": False}
        batch = self.processor(text=conversations, **settings)
        batch.pop("token_type_ids", None)  # Idefics3 does not accept segment IDs.
        prefix = self.processor(text=prompts, **settings)
        labels = batch["input_ids"].clone()
        for i in range(len(examples)):
            length = int(batch["attention_mask"][i].sum())
            prompt_length = int(prefix["attention_mask"][i].sum())
            if length > self.limit:
                raise ValueError(
                    f"Example exceeds max_sequence_length ({length}>{self.limit}); "
                    "shorten data or increase the limit; image tokens are never truncated"
                )
            if prompt_length >= length or not torch.equal(
                prefix["input_ids"][i, :prompt_length], batch["input_ids"][i, :prompt_length]
            ):
                raise ValueError("Chat template does not have a stable assistant-token boundary")
            labels[i, :prompt_length] = -100
        labels[batch["attention_mask"] == 0] = -100
        labels[batch["input_ids"] == self.image_token_id] = -100
        if (labels[:, 1:] != -100).sum(dim=1).min().item() == 0:
            raise ValueError("Every example must contain supervised assistant tokens")
        batch["labels"] = labels
        return batch
