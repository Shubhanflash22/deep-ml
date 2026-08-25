def tree_and_graph_drills(task, *args):

    # task is one of: "lca", "clone", "min_removals", "word_break"

    if task == "lca":
        tree, root, p, q = args

        def lca(node):
            if node is None:
                return None

            if node == p or node == q:
                return node

            left = lca(tree[node][0])
            right = lca(tree[node][1])

            if left and right:
                return node

            return left or right

        return lca(root)

    elif task == "clone":
        graph = args[0]

        cloned = {}

        # Create every node first
        for node in graph:
            cloned[node] = []

        # Connect cloned nodes
        for node in graph:
            for neighbor in graph[node]:
                cloned[node].append(neighbor)

        return {
            node: sorted(cloned[node])
            for node in sorted(cloned)
        }

    elif task == "min_removals":
        s = args[0]

        balance = 0
        removals = 0

        for ch in s:
            if ch == '(':
                balance += 1
            else:
                if balance > 0:
                    balance -= 1
                else:
                    removals += 1

        return removals + balance

    elif task == "word_break":
        s, words = args

        words = set(words)
        dp = [False] * (len(s) + 1)
        dp[0] = True

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break

        return dp[len(s)]

    else:
        raise ValueError("Unknown task")