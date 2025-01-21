from transformers import PreTrainedTokenizer


def infer_response_prefix_ids(
    tokenizer: PreTrainedTokenizer, slice: int = -3
) -> list[int]:
    """
    Infers the token ids of the response prefix from the chat template.
    Works only on tokenizers whose template includes `add_generation_prompt`.

    Args:
        tokenizer (PreTrainedTokenizer): a tokenizer
        slice (int): slice to pick the the last token ids. Useful
                     to avoid `response_template not found` errors
                     in SFTrainer, with tokenizers that tokenize
                     the same token in different ways depending
                     on its left context. See this for more information:
                     https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate
                     Empirically, `-3` worked well for most of the models, but if you
                     get `response_template not found` errors even with a large
                     `max_seq_length`, feel free to change this value.
    Returns:
        list[int]: list of ids of the response prefix.
    """
    dummy_messages = [
        {"role": "system", "content": "dummy"},
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "dummy"},
    ]
    without_gen_prompt = tokenizer.apply_chat_template(
        dummy_messages, tokenize=False, add_generation_prompt=False
    )
    with_gen_prompt = tokenizer.apply_chat_template(
        dummy_messages, tokenize=False, add_generation_prompt=True
    )
    response_prefix = with_gen_prompt[len(without_gen_prompt) :]
    return tokenizer.encode(response_prefix, add_special_tokens=False)[slice:]
