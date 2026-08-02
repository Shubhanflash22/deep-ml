def prepare_eval_input(tokens: list, mode: str, reserved_output: int) -> list:
    """
    Truncate a token list to fit the context window of the given reasoning mode.
    """
    if mode == "non-think":
        C = 8192
    elif mode == "high":
        C = 131072
    elif mode == "max":
        C = 393216
    else:
        C = 0
    L = C - reserved_output
    if L<=0:
        return []
    l = len(tokens)
    if l<=L:
        return tokens
    return tokens[l-L:]