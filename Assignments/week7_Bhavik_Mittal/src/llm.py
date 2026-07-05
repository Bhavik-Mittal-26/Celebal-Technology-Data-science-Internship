"""
Local Language Model Module
------------------------------
Generates grounded answers using retrieved context, entirely locally
via Hugging Face `transformers`. No API key, no paid endpoint.

Default model: google/flan-t5-large
  - Instruction-tuned, good at "answer using this context" style prompts
  - Runs comfortably on CPU (slower) or GPU (fast)
  - ~780M params, public weights, no auth token needed

You can swap MODEL_NAME below for:
  - "google/flan-t5-base"   (smaller/faster, lower quality)
  - "google/flan-t5-xl"     (bigger/slower, higher quality, needs more RAM)
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class LocalLLM:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Loads the tokenizer + seq2seq model directly (rather than via
        transformers' pipeline() task shortcut). transformers v5 removed the
        "text2text-generation" pipeline alias, so this approach is used
        instead — it also works fine on older transformers versions.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _ensure_model(self):
        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        return self.model, self.tokenizer

    def build_prompt(self, question: str, context_chunks: list) -> str:
        """
        Combine retrieved context + question into a single grounded prompt.
        """
        context_text = "\n\n".join(
            f"[Source: {c['source']}] {c['text']}" for c in context_chunks
        )

        prompt = (
            "You are a helpful assistant that answers questions using ONLY the "
            "context provided below. If the answer is not contained in the "
            "context, say 'The document does not contain this information.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly and concisely, grounded strictly in the context above:"
        )
        return prompt

    def generate_answer(self, question: str, context_chunks: list, max_new_tokens: int = 256) -> str:
        if not context_chunks:
            return "No relevant context was found in the document(s) for this question."

        prompt = self.build_prompt(question, context_chunks)

        # Ensure the tokenizer and model are loaded (lazy-init safe)
        model, tokenizer = self._ensure_model()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        # Move all tensor inputs to the desired device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Use faster generation settings by default to reduce latency
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=min(max_new_tokens, 128),
            do_sample=False,
            num_beams=1,
        )

        # Run generation in a worker thread so we can timeout if it hangs
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        def do_generate():
            with torch.no_grad():
                return model.generate(**gen_kwargs)

        gen_timeout = 60  # seconds
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(do_generate)
            try:
                output_ids = future.result(timeout=gen_timeout)
            except TimeoutError:
                return "Generation timed out after {} seconds. Try a smaller model or shorter prompt.".format(
                    gen_timeout
                )

        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip()
