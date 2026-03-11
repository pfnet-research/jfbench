from jfbench.constraints.notation.email_address import _validate_emails_in_text
from jfbench.constraints.notation.email_address import EmailAddressNotationConstraint


def test_email_address_constraint_rejects_wrapped_valid_address_with_strict_pattern() -> None:
    constraint = EmailAddressNotationConstraint()
    text = "お問い合わせは「support@example.com」にお願いします。"
    ok, reason = constraint.evaluate(text)
    assert ok
    assert reason is None


def test_email_address_constraint_rejects_invalid_domain() -> None:
    constraint = EmailAddressNotationConstraint()
    text = "連絡先: user@example をご利用ください。"
    assert not constraint.evaluate(text)[0]


def test_email_address_constraint_rejects_backtick_and_single_quote_wrapped_address() -> None:
    constraint = EmailAddressNotationConstraint()
    text = "チームへの連絡は `support@example.com' まで。"
    ok, reason = constraint.evaluate(text)
    assert ok
    assert reason is None


def test_validate_emails_in_text_reports_missing_email_after_normalization() -> None:
    ok, mismatches = _validate_emails_in_text("support＠example.com", require_presence=True)
    assert ok
    assert mismatches == []


def test_validate_emails_in_text_requires_presence() -> None:
    ok, mismatches = _validate_emails_in_text(
        "ここにはメールがありません。", require_presence=True
    )
    assert not ok
    assert mismatches == [("", "no_email_found")]
