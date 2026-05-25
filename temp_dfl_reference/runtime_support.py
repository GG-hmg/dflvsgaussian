import builtins
import hashlib
import sys


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass


def fix_mojibake_text(text):
    if not isinstance(text, str) or not text:
        return text
    if text.isascii():
        return text
    suspicious = ("èŒ…", "èŽ½", "å¯", "æ°“", "èŒ‚", "å¿™", "èŠ’", "è¬", "èº", "è§", "é‚", "é—†", "é”", "ç»—", "é¨", "é”›", "é", "å§", "å¦«", "ç€¹")
    if not any(token in text for token in suspicious):
        return text
    for source_encoding in ("gbk", "latin1"):
        try:
            return text.encode(source_encoding, errors="strict").decode("utf-8", errors="strict")
        except Exception:
            continue
    return text


def install_safe_print() -> None:
    original_print = builtins.print

    def _safe_print(*args, **kwargs):
        fixed = tuple(fix_mojibake_text(a) if isinstance(a, str) else a for a in args)
        original_print(*fixed, **kwargs)

    builtins.print = _safe_print
    configure_console_output()


def stable_u64_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return seed or 1
