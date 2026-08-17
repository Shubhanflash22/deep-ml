def mask_instruction_loss(batch, pad_token_id, ignore_index=-100):
    L = max(len(item["tokens"]) for item in batch)
    inputs = []
    targets = []

    for item in batch:
        tokens = item["tokens"]
        instruction_len = item["instruction_length"]

        padded = tokens + [pad_token_id] * (L - len(tokens))

        inp = padded[:-1]
        target = padded[1:]

        for i in range(len(target)):
            if i+1<instruction_len:
                target[i] = ignore_index
        
        seen_pad = False

        for i in range(len(target)):
            if target[i] == pad_token_id:
                if not seen_pad:
                    seen_pad = True
                else:
                    target[i] = ignore_index

        inputs.append(inp)
        targets.append(target)


    return {
        "inputs": inputs,
        "targets": targets
    }