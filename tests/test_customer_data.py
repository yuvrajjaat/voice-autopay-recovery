"""Validation of the committed seed data.

These tests are the project's safety net for the "fictional data only" and
"no real payment credentials" requirements. They assert properties of the
actual file, so a careless edit later cannot quietly introduce real-looking
data.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

from app import store

EXPECTED_COUNT = 10

# Field names that would indicate a real payment credential had crept in.
FORBIDDEN_KEYS = {
    "card_number",
    "cardnumber",
    "pan",
    "cvv",
    "cvc",
    "security_code",
    "account_number",
    "routing_number",
    "iban",
    "sort_code",
    "bank_account",
    "token",
    "secret",
    "password",
    "api_key",
}


def test_exactly_ten_customers() -> None:
    assert len(store.list_customers()) == EXPECTED_COUNT


def test_customer_ids_are_unique_and_sequential() -> None:
    ids = [customer.customer_id for customer in store.list_customers()]
    assert len(set(ids)) == EXPECTED_COUNT
    assert ids == [f"CUST-{number:03d}" for number in range(1, EXPECTED_COUNT + 1)]


def test_required_fields_are_present_and_non_empty() -> None:
    """The Customer model enforces the shape; this checks nothing is blank."""
    for customer in store.list_customers():
        assert customer.name.strip()
        assert customer.email.strip()
        assert customer.phone.strip()
        assert customer.timezone.strip()
        assert customer.address.postal_code.strip()
        assert customer.verification.expected_answer.strip()
        assert customer.plan.name.strip()
        assert customer.autopay.primary_method.payment_method_id.strip()
        assert customer.failed_payment.payment_id.strip()
        assert customer.failed_payment.failure_reason_spoken.strip()
        assert customer.scenario.caller_intent.strip()


def test_all_phone_numbers_are_in_the_reserved_fictional_block() -> None:
    """Every number must be NANP 555-0100..555-0199, which cannot be dialled.

    This is the data-level half of the dial-safety design. The other half is
    that the dialler ignores these numbers entirely and only ever calls
    DEMO_PHONE_NUMBER from .env.
    """
    for customer in store.list_customers():
        digits = re.sub(r"\D", "", customer.phone)
        assert len(digits) == 11, f"{customer.customer_id}: expected +1 and 10 digits"
        assert digits.startswith("1"), f"{customer.customer_id}: not a +1 number"
        exchange = digits[4:7]
        line = digits[7:11]
        assert exchange == "555", f"{customer.customer_id}: exchange {exchange} is not 555"
        assert 100 <= int(line) <= 199, (
            f"{customer.customer_id}: line {line} is outside the reserved "
            "555-0100..555-0199 fictional range"
        )


def test_phone_numbers_and_emails_are_unique() -> None:
    customers = store.list_customers()
    assert len({c.phone for c in customers}) == EXPECTED_COUNT
    assert len({c.email for c in customers}) == EXPECTED_COUNT


def test_all_emails_use_the_reserved_example_domain() -> None:
    """example.com is IANA-reserved and cannot receive mail."""
    for customer in store.list_customers():
        assert customer.email.endswith("@example.com"), customer.customer_id


def test_no_payment_credential_fields_exist(seed_path: Path) -> None:
    """Recursively assert that no field name suggests a real credential."""
    document = json.loads(seed_path.read_text(encoding="utf-8"))

    def walk(node: object, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key.lower() not in FORBIDDEN_KEYS, (
                    f"forbidden credential field {key!r} at {path}"
                )
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")

    walk(document)


def test_no_card_number_like_digit_runs(seed_path: Path) -> None:
    """No run of 12+ consecutive digits anywhere in the file.

    Card numbers are 13-19 digits and bank accounts are long too, so banning
    long digit runs catches a pasted credential regardless of field name.
    Phone numbers are 11 digits, which stays under the threshold.
    """
    text = seed_path.read_text(encoding="utf-8")
    long_runs = re.findall(r"\d{12,}", text)
    assert not long_runs, f"suspicious long digit runs found: {long_runs}"


def test_stored_cards_expose_only_brand_and_last_four() -> None:
    for customer in store.list_customers():
        methods = [customer.autopay.primary_method, customer.autopay.backup_method]
        for method in [m for m in methods if m is not None]:
            assert re.fullmatch(r"\d{4}", method.last4), customer.customer_id
            assert method.payment_method_id.startswith("pm_mock_"), customer.customer_id


def test_amounts_are_positive_decimals() -> None:
    """Money must parse as Decimal, never float, and must be positive."""
    for customer in store.list_customers():
        amount = customer.failed_payment.amount
        assert isinstance(amount, Decimal)
        assert amount > Decimal("0")
        assert amount < Decimal("10000"), "demo amounts should stay plausible"


def test_identifiers_are_clearly_mock() -> None:
    for customer in store.list_customers():
        assert customer.failed_payment.payment_id.startswith("pay_mock_")


def test_verification_answer_matches_the_address_on_file() -> None:
    """The postal code the agent checks must be the one on the account."""
    for customer in store.list_customers():
        assert customer.verification.expected_answer == customer.address.postal_code


def test_scenarios_cover_every_conversation_branch() -> None:
    """The ten records must exercise each branch of the planned flow."""
    paths = {customer.scenario.expected_path for customer in store.list_customers()}
    required = {
        "retry_succeeds",
        "retry_succeeds_on_backup",
        "retry_fails",
        "payment_link_sent",
        "retry_scheduled",
        "escalated",
        "verification_failed",
        "do_not_call",
    }
    missing = required - paths
    assert not missing, f"no fictional customer exercises: {sorted(missing)}"


def test_failure_reasons_are_varied() -> None:
    """Several distinct decline codes, so the demo is not one-note."""
    codes = {customer.failed_payment.failure_code for customer in store.list_customers()}
    assert len(codes) >= 6, f"only {len(codes)} distinct failure codes"


def test_suspension_dates_only_where_genuinely_delinquent() -> None:
    """The agent must not be able to imply shutoff where none is scheduled."""
    for customer in store.list_customers():
        payment = customer.failed_payment
        if payment.service_suspension_date is not None:
            assert payment.days_delinquent >= 14, (
                f"{customer.customer_id} has a suspension date but is only "
                f"{payment.days_delinquent} days delinquent"
            )


def test_seed_declares_itself_fictional(seed_path: Path) -> None:
    meta = json.loads(seed_path.read_text(encoding="utf-8"))["_meta"]
    assert meta["fictional"] is True
    assert meta["record_count"] == EXPECTED_COUNT


def test_loading_is_deterministic() -> None:
    """Two loads produce byte-identical data — demos must be repeatable."""
    first = [c.model_dump_json() for c in store.list_customers()]
    second = [c.model_dump_json() for c in store.load_customers(refresh=True).values()]
    assert first == second
