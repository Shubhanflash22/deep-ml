
def term_bounds(terms, ranges):
    """Return the minimum and maximum values of a linear expression."""
    vmin = terms.get("", 0)
    vmax = terms.get("", 0)

    for var, coeff in terms.items():
        if var == "" or coeff == 0:
            continue

        lo, hi = ranges[var]

        if coeff > 0:
            vmin += coeff * lo
            vmax += coeff * (hi - 1)
        else:
            vmin += coeff * (hi - 1)
            vmax += coeff * lo

    return vmin, vmax


def fold_div(terms, c, ranges):
    """Fold divisible terms into the quotient."""
    quotient = {"": 0}
    remainder = {"": 0}

    for var, coeff in terms.items():
        if coeff % c == 0:
            quotient[var] = coeff // c
        else:
            remainder[var] = coeff

    # Remove zero coefficients.
    quotient = {
        k: v for k, v in quotient.items()
        if k == "" or v != 0
    }
    remainder = {
        k: v for k, v in remainder.items()
        if k == "" or v != 0
    }

    # Keep the constant first.
    quotient = {"": quotient.get("", 0),
                **{k: v for k, v in quotient.items() if k != ""}}

    remainder = {"": remainder.get("", 0),
                 **{k: v for k, v in remainder.items() if k != ""}}

    rmin, rmax = term_bounds(remainder, ranges)

    if 0 <= rmin and rmax < c:
        return quotient, None

    return quotient, remainder



def fold_mod(terms, c, ranges):
    remainder = {}

    for var, coeff in terms.items():
        if var != "" and coeff % c != 0:
            if coeff != 0:
                remainder[var] = coeff

    # Constant must always be last.
    constant = terms.get("", 0)

    if constant % c != 0:
        remainder[""] = constant
    else:
        remainder[""] = 0

    rmin, rmax = term_bounds(remainder, ranges)
    exact = 0 <= rmin and rmax < c

    return remainder, exact

def unflatten(shape, flat_terms, ranges):
    """Convert a flattened index expression into multidimensional indices."""
    result = []

    for j, dim in enumerate(shape):
        stride = 1

        for d in shape[j + 1:]:
            stride *= d

        quotient, remainder = fold_div(
            flat_terms, stride, ranges
        )

        modulo, exact = fold_mod(
            quotient, dim, ranges
        )

        if remainder is None and exact:
            result.append(modulo)
        else:
            result.append("div/mod")

    return result
