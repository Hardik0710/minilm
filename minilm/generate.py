import torch


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0,
             top_p=1.0, eos="<|endoftext|>", device=None, min_new_tokens=0):
    """temperature <= 0 means greedy decoding.

    min_new_tokens > 0 blocks <|endoftext|> until that many tokens exist.
    Default 0 gives the normal behaviour (stop when the model emits EOS).
    """
    model.eval()
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt)
    eos_id = tokenizer.bytes_to_id.get(eos.encode("utf-8"))
    new = []
    for _ in range(max_new_tokens):
        ctx = ids[-model.context_length:]                  # last 256 tokens only
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1]                           # last position
        if eos_id is not None and len(new) < min_new_tokens:
            logits = logits.clone()
            logits[eos_id] = float("-inf")                 # block EOS for now
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = torch.cumsum(sp, dim=0)
                # keep the smallest set whose mass reaches top_p
                keep = (cum - sp) < top_p
                sp = sp * keep
                sp = sp / sp.sum()
                nxt = int(si[torch.multinomial(sp, 1)])
            else:
                nxt = int(torch.multinomial(probs, 1))
        if nxt == eos_id:
            break
        ids.append(nxt)
        new.append(nxt)
    return tokenizer.decode(new)