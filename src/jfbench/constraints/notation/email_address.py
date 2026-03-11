import ipaddress
import logging
import re
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class EmailAddressNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        valid, mismatches = _validate_emails_in_text(value, require_presence=True)
        last_reason: str | None = None
        if not valid:
            for email_str, reason in mismatches:
                last_reason = f"[Email Address Notation] Invalid email {email_str!r}: {reason}"
                logger.info(last_reason)
        if valid:
            return True, None
        if last_reason is None:
            last_reason = "[Email Address Notation] Validation failed."
            logger.info(last_reason)
        return False, last_reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には少なくとも1つ、有効な形式のメールアドレスを含めてください。",
            "正しい書式のメールアドレスを回答内に入れてください。",
            "形式が正しいメールアドレス（local@domain）を必ず含めた出力にしてください。",
            "回答中で1件以上の有効なメールアドレスを提示してください。",
            "メールアドレスを記載する場合は仕様に合致した形式を用い、必ず1件以上含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "有効なメールアドレスを1件以上入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "先ほどの出力を修正し、少なくとも1つの形式が正しいメールアドレスを含めてください。"


# --- Extraction ---

_EMAIL_CANDIDATE_RE = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*"
)

# Characters often wrapping emails in text: <...>, (...), […], Japanese brackets, quotes, etc.
_LEFT_WRAP = set("<([{\u300c\u300e\uff08\u3010\u3008\u300a`'\"")
_RIGHT_WRAP = set(">)]}\u300d\u300f\uff09\u3011\u3009\u300b.,;:!?`'\"")


def _normalize(text: str) -> str:
    """NFKC normalization (e.g., full-width -> ASCII)."""
    return unicodedata.normalize("NFKC", text)


def _trim_wrappers(token: str) -> str:
    """Trim common surrounding wrappers/punctuations without touching quotes."""
    # Trim left-side wrappers once
    while token and token[0] in _LEFT_WRAP:
        token = token[1:]
    # Trim right-side wrappers repeatedly (to drop '...).' etc.)
    while token and token[-1] in _RIGHT_WRAP:
        token = token[:-1]
    return token


def _find_email_candidates(text: str) -> list[str]:
    """
    Extract raw email-like tokens from text (after normalization),
    trimming common wrappers/punctuations.
    """
    t = _normalize(text)
    out: list[str] = []
    for m in _EMAIL_CANDIDATE_RE.finditer(t):
        tok = _trim_wrappers(m.group(0))
        if tok:  # keep non-empty
            out.append(tok)
    return out


# --- Validation helpers (close to RFC 5322/1035, but not identical) ---

# dot-atom (ASCII) for unquoted local-part
_DOT_ATOM_RE = re.compile(
    r"^(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*)$"
)

# quoted-string local-part (very simplified)
_QUOTED_LOCAL_RE = re.compile(r'^"([\x20-\x21\x23-\x5B\x5D-\x7E]|\\[\x20-\x7E])*"$')

# domain label (ASCII, after IDNA), 1..63 chars, no leading/trailing '-'
_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def _is_domain_literal(domain: str) -> bool:
    """Allow domain-literals like [127.0.0.1] or [IPv6:...]."""
    if len(domain) >= 2 and domain[0] == "[" and domain[-1] == "]":
        inner = domain[1:-1]
        # Support both IPv4 and IPv6 (with or without 'IPv6:' prefix)
        if inner.lower().startswith("ipv6:"):
            inner = inner[5:]
        try:
            ipaddress.ip_address(inner)
            return True
        except ValueError:
            return False
    return False


def _is_valid_domain(domain: str) -> bool:
    """
    Validate domain name:
      - IDNA encodable (internationalized domain supported)
      - total length <= 253
      - at least 2 labels (e.g., example.com)
      - each label 1..63 chars, ASCII letters/digits/hyphen, no '-' at ends
    """
    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except Exception:
        return False

    if len(ascii_domain) > 253:
        return False

    labels = ascii_domain.split(".")
    if len(labels) < 2:
        return False
    if not all(_LABEL_RE.fullmatch(lbl) for lbl in labels):
        return False
    return True


def _is_valid_local(local: str, allow_quoted: bool = True) -> bool:
    """
    Validate local-part:
      - dot-atom (ASCII)   e.g., name.surname+tag
      - quoted-string (*)  e.g., "name with spaces"  (optional)
      - length <= 64
    """
    if len(local) > 64:
        return False
    if _DOT_ATOM_RE.fullmatch(local):
        # Reject leading/trailing dot or consecutive dots (regex already prevents these)
        return True
    if allow_quoted and _QUOTED_LOCAL_RE.fullmatch(local):
        return True
    return False


def _is_valid_email(addr: str) -> tuple[bool, str]:
    """
    Check whether `addr` is a syntactically valid email address (no DNS lookup).
    Returns (ok, reason_if_invalid).
    """
    # Overall length limit (RFC suggests 254)
    if len(addr) > 254:
        return False, "too_long"

    if "@" not in addr:
        return False, "missing_at"

    local, domain = addr.rsplit("@", 1)
    if not local or not domain:
        return False, "empty_local_or_domain"

    # Domain-literal (e.g., [127.0.0.1]) or domain name
    if _is_domain_literal(domain):
        domain_ok = True
    else:
        domain_ok = _is_valid_domain(domain)

    if not domain_ok:
        return False, "invalid_domain"

    if not _is_valid_local(local):
        return False, "invalid_local"

    return True, ""


# --- Public API ---


def _validate_emails_in_text(
    text: str,
    *,
    require_presence: bool = False,
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Validate all email-like tokens found in `text`.

    Args:
      text: input text (may contain 0..N email addresses).
      require_presence: if True, fail when no email-like token is found.

    Returns:
      (ok, details)
        ok: True iff (a) at least one email exists when require_presence=True, and
                    (b) every found token is syntactically valid.
        details: list of (token, reason) for invalid tokens;
                 if require_presence=True and none found -> [("", "no_email_found")]

    Notes:
      - This is a *syntactic* check (no DNS or mailbox existence check).
      - For production-grade validation with better i18n coverage, consider using
        the `email-validator` package in addition to, or instead of, custom logic.
    """
    candidates = _find_email_candidates(text)
    if require_presence and not candidates:
        return False, [("", "no_email_found")]

    mismatches: list[tuple[str, str]] = []
    for tok in candidates:
        ok, reason = _is_valid_email(tok)
        if not ok:
            mismatches.append((tok, reason))

    return (len(mismatches) == 0, mismatches)
