
def size(expr):
    if not isinstance(expr, tuple):
        return 1

    if expr[0] == "neg":
        return 1 + size(expr[1])

    return 1 + size(expr[1]) + size(expr[2])


def simplify_node(expr):
    if not isinstance(expr, tuple):
        return expr

    op = expr[0]

    if op == "neg":
        a = expr[1]

        if isinstance(a, (int, float)):
            return -a

        if isinstance(a, tuple) and a[0] == "neg":
            return a[1]

        return ("neg", a)

    a, b = expr[1], expr[2]

    # Constant folding
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if op == "add":
            return a + b
        if op == "mul":
            return a * b

    if op == "add":
        # Identity
        if a == 0:
            return b
        if b == 0:
            return a

        # Canonical constants on the right
        if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
            return ("add", b, a)

        # Combine chained constants: (x+c1)+c2
        if isinstance(a, tuple) and a[0] == "add":
            x, c1 = a[1], a[2]
            if isinstance(c1, (int, float)) and isinstance(b, (int, float)):
                return ("add", x, c1 + b)

        # Associativity: x+(y+c) -> (x+y)+c
        if isinstance(b, tuple) and b[0] == "add":
            y, c = b[1], b[2]
            if isinstance(c, (int, float)):
                return ("add", ("add", a, y), c)

        # Associativity: (x+c)+y -> (x+y)+c
        if isinstance(a, tuple) and a[0] == "add":
            x, c = a[1], a[2]
            if isinstance(c, (int, float)) and not isinstance(b, (int, float)):
                return ("add", ("add", x, b), c)

    if op == "mul":
        # Identities and zero
        if a == 0 or b == 0:
            return 0
        if a == 1:
            return b
        if b == 1:
            return a

        # Canonical constants on the right
        if isinstance(a, (int, float)) and not isinstance(b, (int, float)):
            return ("mul", b, a)

        # Combine chained constants: (x*c1)*c2
        if isinstance(a, tuple) and a[0] == "mul":
            x, c1 = a[1], a[2]
            if isinstance(c1, (int, float)) and isinstance(b, (int, float)):
                return ("mul", x, c1 * b)

        # Distribution: (a+b)*c -> a*c+b*c
        if isinstance(a, tuple) and a[0] == "add" and isinstance(b, (int, float)):
            x, y = a[1], a[2]
            return (
                "add",
                ("mul", x, b),
                ("mul", y, b)
            )

    return expr


def simplify(expr):
    if not isinstance(expr, tuple):
        return expr

    op = expr[0]

    # Recursively simplify children first
    if op == "neg":
        current = ("neg", simplify(expr[1]))
    else:
        current = (
            op,
            simplify(expr[1]),
            simplify(expr[2])
        )

    # Apply rewriting until reaching a fixed point
    while True:
        new = simplify_node(current)

        if new == current:
            return current

        current = simplify(new)
