"""
Local smoke-test runner for POST /analyze-ticket.

Drives every case in `SUST_Preli_Sample_Cases.json` against a locally running
server, prints a per-case diff against `expected_output`, and ends with an
aggregate pass/fail summary so the team can spot regressions before submitting.

Usage:
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    python3 test_runner.py            # default: http://localhost:8000
    python3 test_runner.py --base-url http://localhost:9000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
CASES_PATH = Path(__file__).parent / "SUST_Preli_Sample_Cases.json"

# Required output fields per the problem statement. We treat anything outside
# this list (e.g. `confidence`, `reason_codes`) as informational.
REQUIRED_OUTPUT_FIELDS = (
    "ticket_id",
    "relevant_transaction_id",
    "evidence_verdict",
    "case_type",
    "severity",
    "department",
    "agent_summary",
    "recommended_next_action",
    "customer_reply",
    "human_review_required",
)


def _stringify(value: Any) -> str:
    if value is None:
        return "<null>"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _compare_case(case: dict[str, Any], actual: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    """Compare the harness response against the expected output and return a
    structured diff that the report loop can render.
    """
    result: dict[str, Any] = {
        "id": case.get("id", "?"),
        "label": case.get("label", ""),
        "ok": True,
        "checks": [],
        "missing_fields": [],
        "error": error,
        "raw_actual": actual,
    }

    expected = case.get("expected_output") or {}
    if error or actual is None:
        result["ok"] = False
        result["checks"].append(("network", "FAIL", _stringify(error or "no response")))
        return result

    # 1. All required fields present.
    for field in REQUIRED_OUTPUT_FIELDS:
        if field not in actual:
            result["missing_fields"].append(field)
            result["ok"] = False
            result["checks"].append((field, "MISSING", "—"))

    # 2. Strict checks on fields whose values should be deterministic.
    strict_pairs = (
        "ticket_id",
        "relevant_transaction_id",
        "evidence_verdict",
        "case_type",
        "department",
    )
    for field in strict_pairs:
        exp = expected.get(field)
        act = actual.get(field)
        passed = exp == act and field in actual
        result["checks"].append((field, "PASS" if passed else "FAIL", f"actual={_stringify(act)} | expected={_stringify(exp)}"))
        if not passed:
            result["ok"] = False

    # 3. Severity band check — allow equal or adjacent bucket.
    severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    exp_sev = expected.get("severity")
    act_sev = actual.get("severity")
    if exp_sev in severity_order and act_sev in severity_order:
        delta = abs(severity_order[exp_sev] - severity_order[act_sev])
        passed = delta <= 1
        result["checks"].append(("severity", "PASS" if passed else "FAIL", f"actual={act_sev} | expected={exp_sev}"))
        if not passed:
            result["ok"] = False
    else:
        result["checks"].append(("severity", "WARN", f"actual={_stringify(act_sev)} | expected={_stringify(exp_sev)}"))

    # 4. customer_reply safety heuristic — must not contain credential tokens
    # or authority claims. Mirrors the production guardrails so the report
    # catches them even if the server side regresses.
    UNSAFE_TOKENS = ("pin", "otp", "password", "cvv", "card number")
    AUTHORITY_PHRASES = (
        "we will refund",
        "we'll refund",
        "refund confirmed",
        "refund has been",
        "refund processed",
    )
    reply = (actual.get("customer_reply") or "").lower()
    unsafe_hit = next((tok for tok in UNSAFE_TOKENS if tok in reply), None)
    authority_hit = next((p for p in AUTHORITY_PHRASES if p in reply), None)
    if unsafe_hit:
        result["ok"] = False
        result["checks"].append(("customer_reply.safety", "FAIL", f"credential token: {unsafe_hit!r}"))
    elif authority_hit:
        result["ok"] = False
        result["checks"].append(("customer_reply.safety", "FAIL", f"authority phrase: {authority_hit!r}"))
    else:
        result["checks"].append(("customer_reply.safety", "PASS", "no credential / authority hits"))

    # 5. human_review_required — expected is True for the sample cases; warn
    # rather than fail if the model is confident enough to set False.
    exp_hr = expected.get("human_review_required")
    act_hr = actual.get("human_review_required")
    if exp_hr is True and act_hr is False:
        result["checks"].append(("human_review_required", "WARN", "expected True, got False"))
    else:
        result["checks"].append(("human_review_required", "PASS", f"actual={act_hr} | expected={exp_hr}"))

    # 6. Show the agent_summary for human eyeballing.
    result["checks"].append(
        ("agent_summary.text", "INFO", _stringify(actual.get("agent_summary")))
    )

    return result


def _print_case(report: dict[str, Any]) -> None:
    status = "PASS" if report["ok"] else "FAIL"
    print(f"\n[{status}] {report['id']} — {report['label']}")
    if report["error"]:
        print(f"  ! network/parse error: {report['error']}")
    if report["missing_fields"]:
        print(f"  ! missing fields: {', '.join(report['missing_fields'])}")
    for field, verdict, detail in report["checks"]:
        print(f"    - {verdict:<7} {field:<26} {detail}")
    if report["raw_actual"] is not None:
        print("    raw response:")
        for line in json.dumps(report["raw_actual"], ensure_ascii=False, indent=2).splitlines():
            print(f"      {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the public sample cases against a local server.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Server base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--cases", default=str(CASES_PATH), help="Path to the sample-cases JSON file.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.is_file():
        print(f"!! cases file not found: {cases_path}", file=sys.stderr)
        return 2

    with cases_path.open("r", encoding="utf-8") as fh:
        pack = json.load(fh)
    cases = pack.get("cases") or []
    if not cases:
        print("!! no cases found in sample pack", file=sys.stderr)
        return 2

    endpoint = f"{args.base_url.rstrip('/')}/analyze-ticket"
    print(f"Running {len(cases)} case(s) against {endpoint}")
    print(f"Loaded cases from {cases_path}")

    reports: list[dict[str, Any]] = []
    started = time.monotonic()
    for case in cases:
        payload = case.get("input") or {}
        actual: dict[str, Any] | None = None
        error: str | None = None
        try:
            response = requests.post(endpoint, json=payload, timeout=args.timeout)
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:300]}"
            else:
                actual = response.json()
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
        reports.append(_compare_case(case, actual, error))
        _print_case(reports[-1])

    elapsed = time.monotonic() - started
    passed = sum(1 for r in reports if r["ok"])
    failed = len(reports) - passed
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} passed / {failed} failed of {len(reports)} in {elapsed:.2f}s")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())