import random
from dataclasses import dataclass
from typing import List, Tuple

# Level configuration per operator: (min_value, max_value, max_operators_in_expression)
LEVELS = {
    1: {
        '+': (-100, 100, 2),
        '-': (-100, 100, 2),
        '*': (-10, 10, 2),
        '/': (-10, 10, 2),
    },
    2: {
        '+': (-500, 500, 2),
        '-': (-500, 500, 2),
        '*': (-20, 20, 2),
        '/': (-20, 20, 2),
    },
    3: {
        '+': (-1000, 1000, 4),
        '-': (-1000, 1000, 4),
        '*': (-30, 30, 4),
        '/': (-30, 30, 4),
    },
    # Levels 4-6: handled by dedicated generator (mul/div only); keep placeholders for compatibility
    4: {
        '+': (0, 0, 2),
        '-': (0, 0, 2),
        '*': (0, 0, 2),
        '/': (0, 0, 2),
    },
    5: {
        '+': (0, 0, 3),
        '-': (0, 0, 3),
        '*': (0, 0, 3),
        '/': (0, 0, 3),
    },
    6: {
        '+': (0, 0, 3),
        '-': (0, 0, 3),
        '*': (0, 0, 3),
        '/': (0, 0, 3),
    },
}

OPERATORS = ['+', '-', '*', '/']


def clamp_range(a: int, b: int) -> Tuple[int, int]:
    return (min(a, b), max(a, b))


def rand_inclusive(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def pick_operand_for_op(level: int, op: str, exclude_zero: bool = False) -> int:
    lo, hi, _ = LEVELS[level][op]
    # Handle cases where range could be (0,0) and exclude_zero is True
    if exclude_zero and lo <= 0 <= hi:
        # choose from negative or positive range excluding 0
        candidates = []
        if lo < 0:
            candidates.extend(range(lo, 0))
        if hi > 0:
            candidates.extend(range(1, hi + 1))
        if not candidates:
            return 1  # safe fallback
        return random.choice(candidates)
    # Normal selection
    v = rand_inclusive(lo, hi)
    if exclude_zero and v == 0:
        # simple retry avoiding 0
        if v + 1 <= hi:
            return v + 1
        elif v - 1 >= lo:
            return v - 1
        else:
            return 1
    return v


def allowed_divisors_for_chain(chain_n: int, chain_d: int, level: int) -> List[int]:
    """
    Return candidate divisors within the level '/'' range such that dividing the current
    multiplicative chain by that divisor keeps the simplified denominator allowed.
    Levels 1-2: denominator must be 1 (integer). Level 3: denominator can be 1 or 2.
    Additional constraints: divisor must be positive (>0) to avoid dividing by negative numbers.
    """
    lo, hi, _ = LEVELS[level]['/']
    ds: List[int] = []
    for d in range(max(1, lo), max(1, hi) + 1):  # only positive divisors (>0)
        if d == 0:
            continue
        # After dividing by d: new fraction = chain_n / (chain_d * d)
        from math import gcd
        nd = chain_d * d
        g = gcd(abs(chain_n), abs(nd)) if chain_n != 0 else abs(nd)
        sd = nd // g if g else nd
        sd = abs(sd)
        if level in (1, 2):
            if sd == 1:
                ds.append(d)
        else:  # level 3
            if sd in (1, 2):
                ds.append(d)
    return ds


def compute_expression_value(tokens: List[str]) -> Tuple[int, int]:
    """
    Evaluate with standard precedence using rational arithmetic restricted to denominators 1 or 2.
    Returns a tuple (numerator, denominator) reduced, with denominator in {1,2} where possible.
    """
    from math import gcd

    def reduce_frac(n: int, d: int) -> Tuple[int, int]:
        if d < 0:
            n, d = -n, -d
        g = gcd(abs(n), abs(d)) if n != 0 else abs(d)
        if g:
            n //= g
            d //= g
        # Further limit denominator to 2 if possible (we only allow halves as non-integers)
        if d not in (1, 2):
            # Try to reduce to denominator 2 if divisible
            if d % 2 == 0:
                factor = d // 2
                if n % factor == 0:
                    n //= factor
                    d = 2
        return n, d

    i = 0
    out: List[str] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('*', '/'):
            raise ValueError('Expression cannot start with operator')
        # Start accumulator as fraction
        acc_n = int(tok)
        acc_d = 1
        i += 1
        # Consume multiplicative chain
        while i < len(tokens) and tokens[i] in ('*', '/'):
            op = tokens[i]
            rhs = int(tokens[i + 1])
            if op == '*':
                acc_n *= rhs
            else:
                # rational division
                acc_d *= rhs
            acc_n, acc_d = reduce_frac(acc_n, acc_d)
            i += 2
        out.append(f"{acc_n}/{acc_d}")
        if i < len(tokens):
            out.append(tokens[i])  # '+' or '-'
            i += 1
    # Now reduce additions/subtractions
    # Initialize
    first_n, first_d = map(int, out[0].split('/'))
    total_n, total_d = first_n, first_d
    j = 1
    while j < len(out):
        op = out[j]
        val_n, val_d = map(int, out[j + 1].split('/'))
        if op == '+':
            total_n = total_n * val_d + val_n * total_d
            total_d = total_d * val_d
        else:
            total_n = total_n * val_d - val_n * total_d
            total_d = total_d * val_d
        total_n, total_d = reduce_frac(total_n, total_d)
        j += 2
    return reduce_frac(total_n, total_d)


def _generate_mul_div_only(level: int) -> Tuple[str, Tuple[int, int]]:
    """
    Levels 4–6: Only '*' and '/' operators. Final result must be an integer.
    Additional constraints per level:
      - Level 4: exactly 2 operators; sum of all operands (first number and all following) <= 20
      - Level 5: exactly 3 operators; sum of all operands <= 30
      - Level 6: exactly 3 operators; sum of all operands <= 40
    All operands are positive integers >= 2 to keep tasks meaningful.
    """
    op_counts = {4: 2, 5: 3, 6: 3}
    sum_limits = {4: 20, 5: 30, 6: 40}
    if level not in op_counts:
        raise ValueError("Invalid level for mul/div only generator")

    ops_needed = op_counts[level]
    limit = sum_limits[level]

    # We'll attempt a number of retries to satisfy constraints
    for _attempt in range(200):
        tokens: List[str] = []
        # Choose starting operand between 2 and limit-ops_needed (reserve room)
        start_max = max(2, limit - 2 * ops_needed)
        first = rand_inclusive(2, max(2, min(9, start_max)))
        tokens.append(str(first))
        ssum = first
        value = first
        ok = True
        for i in range(ops_needed):
            # Prefer multiplication early to build factors for later exact divisions
            if i < ops_needed - 1:
                op = random.choice(['*', '*', '/'])
            else:
                op = random.choice(['*', '/'])

            # Compute candidates depending on op
            if op == '*':
                # possible factors: 2..(limit - ssum)
                max_add = limit - ssum
                if max_add < 2:
                    # try division instead
                    op = '/'
                else:
                    candidates = list(range(2, max_add + 1))
                    # Avoid too-large jumps; keep variety
                    random.shuffle(candidates)
                    # pick one
                    chosen = None
                    for c in candidates:
                        # Multiplying always keeps integer; accept
                        chosen = c
                        break
                    if chosen is None:
                        op = '/'
                    else:
                        tokens.append(op)
                        tokens.append(str(chosen))
                        ssum += chosen
                        value *= chosen
                        continue
            if op == '/':
                # divisors of current value between 2 and (limit - ssum)
                max_add = limit - ssum
                if max_add < 2:
                    ok = False
                    break
                # find proper divisors of value (>=2)
                divs: List[int] = []
                # To keep it fast, sample candidates up to max_add
                upper = max_add
                for d in range(2, upper + 1):
                    if value % d == 0:
                        divs.append(d)
                if not divs:
                    # cannot divide; try to switch to multiplication if possible
                    max_add = limit - ssum
                    if max_add >= 2:
                        op = '*'
                        c = rand_inclusive(2, max_add)
                        tokens.append(op)
                        tokens.append(str(c))
                        ssum += c
                        value *= c
                        continue
                    ok = False
                    break
                # avoid trivial 1 (already excluded) and huge divisors to keep interesting
                random.shuffle(divs)
                chosen = divs[0]
                tokens.append('/')
                tokens.append(str(chosen))
                ssum += chosen
                value //= chosen
                continue
        if ok:
            expr = ' '.join(tokens)
            return expr, (value, 1)
    # Fallback simple expression if all attempts fail (should be rare)
    base = 2
    if ops_needed == 2:
        expr = f"{base} * 2 / 2"
        return expr, (base, 1)
    else:
        expr = f"{base} * 2 * 2 / 2"
        return expr, (base * 2, 1)


def generate_expression(level: int) -> Tuple[str, Tuple[int, int]]:
    """
    Generate a random expression and its result for the given level.
    Rules:
      - Levels 1–3: Mixed operators allowed, with standard precedence and per-level rules.
      - Levels 4–6: Only '*' and '/', integer results; constraints per level.
    Returns: (expression_str, (numerator, denominator)) reduced; denominator is 1 or 2.
    """
    if level not in LEVELS:
        raise ValueError('Invalid level')

    if level in (4, 5, 6):
        return _generate_mul_div_only(level)

    # Decide number of operators (1..max)
    max_ops = max(v[2] for v in LEVELS[level].values())
    num_ops = rand_inclusive(1, max_ops)

    tokens: List[str] = []

    # We'll build alternating numbers and operators, ensuring integer division
    # Strategy: construct expression as sequence of additive terms; each term is a multiplicative chain
    remaining_ops = num_ops

    # Generate first number; choose a moderate range using '+' config, excluding zero
    first_num = pick_operand_for_op(level, '+', exclude_zero=True)
    tokens.append(str(first_num))

    # Keep track whether last operator added was '+' or '-' to start new mult chain
    last_additive = True  # start of a term

    while remaining_ops > 0:
        # Choose next operator. If we are at term start, we can choose + or - or * or /.
        # But if we choose '/', we must ensure exact division of the preceding accumulative factor (current term so far).
        # We'll bias toward all operators equally, but enforce constraints (no * or / by negative operands).
        op = random.choice(OPERATORS)

        # Determine current multiplicative chain segment to check constraints
        start = 0
        for m in range(len(tokens) - 2, -1, -1):
            if tokens[m] in ('+', '-'):
                start = m + 1
                break
        chain_tokens_for_check = tokens[start:]
        if chain_tokens_for_check and chain_tokens_for_check[-1] in ('*', '/'):
            chain_tokens_for_check = chain_tokens_for_check[:-1]
        chain_n, chain_d = compute_expression_value(chain_tokens_for_check) if chain_tokens_for_check else (int(tokens[-1]), 1)

        # If choosing '*' or '/', forbid when left operand (current chain) is negative
        if op in ('*', '/') and (chain_n * (1 if chain_d > 0 else -1)) < 0:
            op = random.choice(['+', '-'])

        # Enforce operand-range rule for multiplicative ops: the left operand of '*' or '/'
        # must also be within the configured range for that operator (by absolute value).
        if op in ('*', '/'):
            max_abs = max(abs(LEVELS[level][op][0]), abs(LEVELS[level][op][1]))
            if abs(chain_n) > max_abs * abs(chain_d):
                # Too large to be a valid multiplicand/dividend for this level → switch to + or -
                op = random.choice(['+', '-'])

        # Ensure we don't start a multiplicative chain with division by zero and keep exact/allowed division
        if op == '/':
            # Compute allowed divisors under current chain and level
            ds = allowed_divisors_for_chain(chain_n, chain_d, level)
            if not ds:
                # If no allowed division, fallback to multiplication this step
                op = '*'
        tokens.append(op)

        # Now pick operand based on operator
        if op == '/':
            # Recompute chain value as above to choose a divisor
            start = 0
            for m in range(len(tokens) - 2, -1, -1):
                if tokens[m] in ('+', '-'):
                    start = m + 1
                    break
            # tokens currently end with '/', remove it to evaluate current chain
            chain_tokens = tokens[start:]
            if chain_tokens and chain_tokens[-1] in ('*', '/'):
                chain_tokens = chain_tokens[:-1]
            chain_n, chain_d = compute_expression_value(chain_tokens)
            choices = allowed_divisors_for_chain(chain_n, chain_d, level)
            # Avoid 1 too often to keep it interesting; only positive divisors are returned
            choices = [d for d in choices if d != 1] or choices
            operand = random.choice(choices) if choices else 1
        elif op == '*':
            # Multiply by positive, non-zero
            operand = abs(pick_operand_for_op(level, '*', exclude_zero=True))
            if operand == 0:
                operand = 1
        elif op == '+':
            operand = pick_operand_for_op(level, '+', exclude_zero=True)
        else:  # '-'
            operand = pick_operand_for_op(level, '-', exclude_zero=True)
        # Normalize additive operator with negative operand: '+ -N' -> '- N', '- -N' -> '+ N'
        if op in ('+', '-') and isinstance(operand, int) and operand < 0:
            # flip last operator token
            tokens[-1] = '-' if op == '+' else '+'
            operand = -operand
        # Avoid division by zero just in case and ensure divisor positive
        if op == '/' and (operand == 0 or operand < 0):
            operand = 1
        tokens.append(str(operand))

        remaining_ops -= 1

    # Compute final value as fraction
    result = compute_expression_value(tokens)
    expr = ' '.join(tokens)
    return expr, result


def tokens_to_spoken(tokens: List[str]) -> str:
    words = []
    for t in tokens:
        if t == '+':
            words.append('plus')
        elif t == '-':
            words.append('minus')
        elif t == '*':
            words.append('times')
        elif t == '/':
            words.append('divided by')
        else:
            words.append(t)
    return ' '.join(words)
