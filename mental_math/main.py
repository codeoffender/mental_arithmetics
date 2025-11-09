import argparse
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mental_math.generator import generate_expression
from mental_math.tts import speak, operator_to_word


def _emit_beep() -> None:
    """Emit a short cross-platform beep to indicate 5 seconds remaining."""
    try:
        # Windows: use winsound for a reliable beep
        import platform as _plat
        if _plat.system().lower() == 'windows':
            try:
                import winsound  # type: ignore
                winsound.Beep(880, 200)  # 880 Hz for 200 ms
                return
            except Exception:
                pass
        # Fallback: terminal bell character (may be disabled in some terminals)
        sys.stdout.write('\a')
        sys.stdout.flush()
    except Exception:
        # Last resort: ignore errors, as beep is non-critical
        pass


def read_input_with_timeout(prompt: str, timeout_s: int) -> Tuple[bool, Optional[str]]:
    """
    Returns (got_answer, text). got_answer False means timed out.
    """
    result: List[Optional[str]] = [None]

    def _reader():
        try:
            result[0] = input(prompt)
        except EOFError:
            result[0] = None

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        return False, None
    return True, result[0]


@dataclass
class TaskResult:
    expression: str
    correct: Tuple[int, int]  # (numerator, denominator), denominator in {1,2}
    user_answer_raw: Optional[str]
    is_timeout: bool


LEVEL_HELP = (
    "Level 1: +, -, *, / with ranges: +/-(−100..100), */÷(−10..10), up to 2 operators per expression — division results are integers.\n"
    "Level 2: +, -, *, / with ranges: +/-(−500..500), */÷(−20..20), up to 2 operators per expression — division results are integers.\n"
    "Level 3: +, -, *, / with ranges: +/-(−1000..1000), */÷(−30..30), up to 4 operators per expression — division may yield halves (x.5).\n"
    "Level 4: only * and ÷, exactly 2 operators; all operands positive; integer results; sum of operands ≤ 20.\n"
    "Level 5: only * and ÷, exactly 3 operators; all operands positive; integer results; sum of operands ≤ 30.\n"
    "Level 6: only * and ÷, exactly 3 operators; all operands positive; integer results; sum of operands ≤ 40.\n"
)


def run_session(level: int, num_tasks: int, time_per_task: int, tts_enabled: bool, show_expr: bool) -> List[TaskResult]:
    results: List[TaskResult] = []

    for idx in range(1, num_tasks + 1):
        expr, correct = generate_expression(level)

        # Speak expression (German operators)
        if tts_enabled:
            tokens = expr.split()
            spoken_parts: List[str] = []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                # Handle sequences like '+ -5' or '- -5' to avoid double minus in speech
                if tok in ['+','-'] and i + 1 < len(tokens):
                    nxt = tokens[i+1]
                    if nxt.startswith('-') and nxt[1:].isdigit():
                        # Flip additive operator with following negative number for speech only
                        # '+ -5' -> 'minus 5', '- -5' -> 'plus 5'
                        flipped = 'plus' if tok == '-' else 'minus'
                        spoken_parts.append(flipped)
                        spoken_parts.append(nxt[1:])  # absolute value
                        i += 2
                        continue
                # If token is a negative number (including at the beginning or after * /), speak it as 'minus <abs>'
                if (tok.startswith('-') and tok[1:].isdigit()):
                    spoken_parts.append('minus')
                    spoken_parts.append(tok[1:])
                    i += 1
                    continue
                if tok in ['+','-','*','/']:
                    spoken_parts.append(operator_to_word(tok))
                else:
                    spoken_parts.append(tok)
                i += 1
            spoken = ' '.join(spoken_parts)
            _ = speak(spoken)
        if show_expr:
            print(f"Task {idx}: {expr} = ?")

        # Schedule a 5-seconds-left beep for this task
        cancel_beep = threading.Event()
        if time_per_task > 5:
            def _beeper():
                wait_s = max(0, time_per_task - 5)
                # Wait for 'wait_s' seconds unless canceled; if not canceled, emit beep
                if not cancel_beep.wait(wait_s):
                    _emit_beep()
            threading.Thread(target=_beeper, daemon=True).start()

        got, text = read_input_with_timeout("Your answer: ", time_per_task)
        # Stop/cancel any pending beep for this task
        cancel_beep.set()

        if not got:
            print("Time's up!")
            results.append(TaskResult(expr, correct, None, True))
            continue
        results.append(TaskResult(expr, correct, text, False))

    return results


def summarize(results: List[TaskResult]) -> None:
    def parse_user_answer(text: str) -> Optional[Tuple[int, int]]:
        s = text.strip().replace(',', '.')
        # Accept integers or x.5
        try:
            if '.' in s:
                f = float(s)
                # Only accept halves
                frac = f - int(f)
                if abs(frac - 0.5) < 1e-9:
                    n = int(floor := int(f))
                    # For negative numbers with .5, example -2.5: int(-2.5) == -2 in Python, but -2.5 = -(2) - 0.5
                    # We'll reconstruct using round toward zero logic
                    if f >= 0:
                        return (int(floor) * 2 + 1, 2)
                    else:
                        # e.g., -2.5 -> (-2*2 -1)/2 = -5/2
                        return (int(floor) * 2 - 1, 2)
                else:
                    return None
            else:
                iv = int(s)
                return (iv, 1)
        except Exception:
            return None

    def format_frac(n: int, d: int) -> str:
        if d == 1:
            return str(n)
        # Denominator 2 -> show .5
        if n >= 0:
            return f"{n//2}.5" if n % 2 else str(n//2)
        else:
            # For negative half: e.g., -5/2 -> -2.5
            q, r = divmod(abs(n), 2)
            sign = '-'
            if r == 0:
                return f"{sign}{q}"
            else:
                return f"{sign}{q}.5"

    total = len(results)
    correct_n = 0
    wrong_details: List[str] = []

    for r in results:
        corr_n, corr_d = r.correct
        user_frac: Optional[Tuple[int, int]] = None
        if not r.is_timeout and r.user_answer_raw is not None:
            user_frac = parse_user_answer(r.user_answer_raw)
        if user_frac is not None and user_frac == (corr_n, corr_d):
            correct_n += 1
        else:
            given = "<no answer> (timeout)" if r.is_timeout else (r.user_answer_raw if r.user_answer_raw is not None else "<no answer>")
            wrong_details.append(f"- {r.expression} = {format_frac(corr_n, corr_d)} | you: {given}")

    pct = (correct_n / total * 100.0) if total else 0.0
    print()
    print(f"Result: {correct_n}/{total} correct ({pct:.0f}%).")
    if wrong_details:
        print("Wrong or missed tasks:")
        print("\n".join(wrong_details))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Mental arithmetic trainer (spoken tasks)")
    p.add_argument("--level", "-l", type=int, choices=[1, 2, 3, 4, 5, 6], default=1, help="Difficulty level (1..6)")
    p.add_argument("--tasks", "-n", type=int, default=20, help="Number of tasks per run (default 20)")
    p.add_argument("--time", "-t", type=int, default=30, help="Seconds per task (default 30)")
    p.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    p.add_argument("--show", action="store_true", help="Also print the expression (by default only spoken)")
    p.add_argument("--about-levels", action="store_true", help="Show level details and exit")
    args = p.parse_args(argv)

    if args.about_levels:
        print(LEVEL_HELP)
        return 0

    print(f"Starting mental math trainer — Level {args.level}, {args.tasks} tasks, {args.time}s per task.")
    if args.no_tts:
        print("Note: TTS disabled; expressions will not be read aloud.")

    results = run_session(level=args.level, num_tasks=args.tasks, time_per_task=args.time, tts_enabled=not args.no_tts, show_expr=args.show)
    summarize(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
