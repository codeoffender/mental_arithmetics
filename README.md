### Mental Arithmetic Trainer (Spoken)

A simple command‑line trainer for mental arithmetic with spoken tasks, per‑task time limits, three difficulty levels, mixed operators with standard precedence, and a clear summary at the end.

Key features:
- Tasks are read out loud (TTS). Printing can be optionally enabled.
- 30 seconds per task by default (configurable).
- 20 tasks per run by default (configurable).
- Mixed operators allowed (e.g., `10 + 5 * 3`). Standard mathematical precedence applies.
- Three difficulty levels match your specification:
  - Level 1: `+`/`-` operands from −100..100; `*`/`/` operands from −10..10; up to 2 operators.
  - Level 2: `+`/`-` operands from −500..500; `*`/`/` operands from −20..20; up to 3 operators.
  - Level 3: `+`/`-` operands from −1000..1000; `*`/`/` operands from −30..30; up to 4 operators.
- Text-to-speech is spoken in German by default (uses a German voice if available on your system).
- Division results are integers on Levels 1–2; on Level 3, divisions may yield half values (x.5), e.g., 7 ÷ 2 = 3.5.
- End‑of‑run overview: percentage correct and a list of wrong/timeout tasks showing the expression, correct answer, and your given answer.

---

#### Quick start

Requirements: Python 3.8+.

Optional for TTS (text‑to‑speech):
- Cross‑platform: `pip install pyttsx3`
- macOS: built‑in `say` command works out of the box
- Linux: `espeak` (e.g., `sudo apt install espeak`)
- Windows: PowerShell SAPI.SpVoice works by default on most systems

Run from the project root:

```
python3 -m mental_math --about-levels
```

Start a session (defaults: level 1, 20 tasks, 30s per task, TTS enabled):

```
python3 -m mental_math
```

Common options:
- Choose difficulty level 1–3:
  ```
  python3 -m mental_math -l 2
  ```
- Change number of tasks (default 20):
  ```
  python3 -m mental_math -n 20
  ```
- Change time per task in seconds (default 30):
  ```
  python3 -m mental_math -t 45
  ```
- Disable TTS (not recommended, but available):
  ```
  python3 -m mental_math --no-tts
  ```
- Also print each expression (in addition to speaking it):
  ```
  python3 -m mental_math --show
  ```

You will be prompted to type answers as integers or halves (x.5). You can use a dot or comma as decimal separator (e.g., 3.5 or 3,5). Each task times out individually; timeouts are recorded in the summary.

---

#### Notes on content generation
- Expressions use mixed operators with standard precedence (multiply/divide before add/subtract, left‑to‑right within the same precedence).
- Operand ranges follow the operator being applied (e.g., multiplication/division use the smaller ranges listed above per level), and division operands are chosen as divisors to keep results integral.

---

#### Project structure
- `mental_math/main.py` — CLI entry point and session loop
- `mental_math/generator.py` — expression generator and evaluator
- `mental_math/tts.py` — text‑to‑speech abstraction with multiple fallbacks
- `mental_math/__main__.py` — enables `python -m mental_math`

---

#### Troubleshooting
- No speech output: install a TTS backend (e.g., `pip install pyttsx3`, or `espeak` on Linux). You can also run with `--show` to display the expression.
- On Windows, ensure PowerShell is available in `PATH`. On Linux, install `espeak`. On macOS, `say` should work by default.

---

#### License
MIT (or adapt as you wish).
