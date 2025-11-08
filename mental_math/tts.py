import platform
import subprocess
import sys
from typing import Optional

try:
    import pyttsx3  # type: ignore
except Exception:
    pyttsx3 = None  # type: ignore


def _pyttsx3_set_german_voice(engine: "pyttsx3.Engine") -> None:  # type: ignore[name-defined]
    try:
        voices = engine.getProperty("voices")
        # Prefer voices explicitly marked as German
        def is_de(v) -> bool:
            lid = (getattr(v, "id", "") or "").lower()
            name = (getattr(v, "name", "") or "").lower()
            langs = getattr(v, "languages", []) or []
            langs = [str(x).lower() for x in langs]
            markers = ["de", "de_de", "de-de", "german", "deutsch"]
            return any(m in lid or m in name for m in markers) or any("de" in l for l in langs)
        german = [v for v in voices if is_de(v)]
        if german:
            engine.setProperty("voice", german[0].id)
    except Exception:
        pass


def speak(text: str) -> bool:
    """
    Speak the given text aloud in German using the best available TTS backend.
    Returns True if speech was attempted, False otherwise.
    """
    # Try pyttsx3 first (offline, cross-platform)
    if pyttsx3 is not None:
        try:
            engine = pyttsx3.init()
            _pyttsx3_set_german_voice(engine)
            try:
                # Slightly slower than default (~200 wpm). Use ~160 for a comfortable pace.
                engine.setProperty("rate", 160)
            except Exception:
                pass
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception:
            pass

    # Fallback to platform-specific commands
    system = platform.system().lower()
    try:
        if system == "darwin":  # macOS
            # Try a known German voice first, then fallback
            for voice in ("Anna", "Marlene", "Markus"):
                try:
                    subprocess.run(["say", "-v", voice, "-r", "160", "--", text], check=True)
                    return True
                except Exception:
                    continue
            subprocess.run(["say", "-r", "160", "--", text], check=True)
            return True
        elif system == "linux":
            # Prefer espeak with German voice and slightly slower rate
            subprocess.run(["espeak", "-v", "de", "-s", "160", "--", text], check=True)
            return True
        elif system == "windows":
            # Use PowerShell SAPI.SpVoice with German voice if available, slower rate
            sanitized = text.replace("`", "").replace("\"", "\\\"")
            ps = (
                "Add-Type -AssemblyName System.speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$voices = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq 'de' }; "
                "if ($voices.Count -gt 0) { $s.SelectVoice($voices[0].VoiceInfo.Name) } "
                "$s.Rate = -1; "
                f"$s.Speak(\"{sanitized}\");"
            )
            subprocess.run(["powershell", "-Command", ps], check=True)
            return True
    except Exception:
        pass

    # As last resort, write to stderr that speech is unavailable
    sys.stderr.write("[WARN] Text-to-speech unavailable. Install 'pyttsx3' or 'espeak', or use macOS 'say'.\n")
    return False


def operator_to_word(op: str) -> str:
    # German operator words
    return {
        '+': 'plus',
        '-': 'minus',
        '*': 'mal',
        '/': 'geteilt durch',
    }.get(op, op)
