SYSTEM_PROMPT = (
    "You are a careful legal assistant. Answer the question using only the contract excerpts provided. "
    "If the excerpts do not contain the answer, say so."
)

# Used ONLY while writing the self-supervised training responses (never in the training prompt).
STYLE_HINTS = {
    "plain": "Explain the answer in plain, simple language that a non-lawyer can follow.",
    "technical": "Answer precisely using the contract's own legal terminology and cite the relevant clause wording.",
    "concise": "Answer in one or two short sentences.",
    "cautious": "Answer carefully, point out risks, conditions or exceptions the reader should watch for.",
}


def format_context(chunk_texts):
    return "\n\n".join(f"[Excerpt {i + 1}]\n{t.strip()}" for i, t in enumerate(chunk_texts))


def build_prompt_text(tokenizer, chunk_texts, question, style=None):
    user = f"Contract excerpts:\n{format_context(chunk_texts)}\n\nQuestion: {question}"
    if style is not None:
        user += f"\n\nInstruction: {STYLE_HINTS[style]}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_ids(tokenizer, chunk_texts, question, max_tokens, style=None):
    """Tokenise the prompt; drop the lowest-ranked excerpts until it fits in max_tokens."""
    texts = list(chunk_texts)
    while True:
        text = build_prompt_text(tokenizer, texts, question, style)
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) <= max_tokens or len(texts) <= 1:
            return ids[-max_tokens:] if len(ids) > max_tokens else ids
        texts = texts[:-1]
