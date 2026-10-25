"""Credential-free regressions for the independently installed projection."""

import json
import shutil
from decimal import Decimal

import pytest

from public_replay.analysis import check_status, comparison, measured_attempts
from public_replay.bundle import REQUIRED, Bundle, canonical, digest, safe_path
from public_replay.sanitize import redact
from public_replay.usage import normalize_usage, numeric


def make_bundle(root, updates=None):
    root.mkdir()
    values = {name: {} for name in REQUIRED}
    values.update({"attempts.json": [], "requests.json": []})
    values.update(updates or {})
    entries = {}
    for name, value in values.items():
        data = canonical(value)
        (root / name).write_bytes(data)
        entries[name] = digest(data)
    (root / "manifest.json").write_bytes(
        canonical(
            {
                "schema_version": 1,
                "files": entries,
                "content_id": digest(canonical(entries)),
            }
        )
    )
    return Bundle(root)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": None, "completion_tokens": 3},
        {"prompt_tokens": 3, "completion_tokens": None},
        {"prompt_tokens": True, "completion_tokens": False, "cost": True},
    ],
)
def test_unknown_is_not_zero_or_complete(usage):
    result = normalize_usage([{"response": {"usage": usage}}])
    assert not result["token_usage_complete"]
    assert result["cost_usd"] is None
    assert result["cost_usd_known"] == 0


def test_zero_paid_malformed_and_reasoning_subset():
    records = [
        {
            "response": {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 3,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                    "cost": 0,
                }
            }
        },
        {
            "response": {
                "error": "malformed tool response",
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "cost": 0.2},
            }
        },
    ]
    result = normalize_usage(records)
    assert result["input_tokens"] == 4
    assert result["output_tokens"] == 8  # reasoning already included
    assert result["reasoning_tokens"] is None
    assert result["reasoning_tokens_known"] == 2
    assert result["cost_usd"] == 0.2
    assert result["cost_usd_complete"]
    assert normalize_usage(records, expected_calls=3)["cost_usd"] is None
    assert normalize_usage(records, calls_complete=False)["cost_usd"] is None


@pytest.mark.parametrize(
    "response", [None, [], "invalid", {"usage": None}, {"provider_response": None}]
)
def test_missing_response(response):
    result = normalize_usage([{"response": response}])
    assert result["input_tokens"] is None


@pytest.mark.parametrize(
    "value", [True, False, None, -1, float("nan"), float("inf"), "1"]
)
def test_invalid_measurements(value):
    assert not numeric(value)


def test_numbers_and_count_bounds():
    assert numeric(Decimal("0")) and numeric(0, tokens=True)
    assert not numeric(0.5, tokens=True)
    for count in [True, 1.5, -1]:
        with pytest.raises(ValueError):
            normalize_usage([], expected_calls=count)


def test_manifest_tamper_missing_and_paths(tmp_path):
    bundle = make_bundle(tmp_path / "bundle")
    assert bundle.optional_audit()["status"] == "not bundled"
    for path in ["../escape", "/escape", "dir/../../escape", "dir\\escape"]:
        with pytest.raises(ValueError):
            safe_path(bundle.root, path)
    (bundle.root / "attempts.json").write_text("[]")
    with pytest.raises(ValueError, match="validation failed: attempts.json"):
        Bundle(bundle.root)
    (bundle.root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest missing"):
        Bundle(bundle.root)


def test_symlinks_and_relocation(tmp_path):
    bundle = make_bundle(tmp_path / "old")
    shutil.copytree(bundle.root, tmp_path / "new root with spaces")
    assert Bundle(tmp_path / "new root with spaces").manifest == bundle.manifest
    (bundle.root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        safe_path(bundle.root, "escape/anything")


def test_redaction_nested_free_text():
    sentinels = [
        "sk-SENTINEL123456789",
        "SENTINEL_PASSWORD",
        "SENTINEL_BEARER",
        "SENTINEL_PRIVATE",
        "SENTINEL_URL",
    ]
    sample = {
        "nested": [{"OPENAI_API_KEY": sentinels[0], "text": "Bearer " + sentinels[2]}],
        "text": "password="
        + sentinels[1]
        + " https://name:"
        + sentinels[4]
        + "@example.invalid/path",
        "block": "-----BEGIN PRIVATE KEY-----\n"
        + sentinels[3]
        + "\n-----END PRIVATE KEY-----",
        "home": "/" + "Users/fixture/credential",
    }
    result = json.dumps(redact(sample))
    assert all(secret not in result for secret in sentinels)
    assert "/Users/" not in result
    assert "[REDACTED]" in result


def test_check_status():
    assert check_status(1, 1, present=True, ran=True) == "pass"
    assert check_status(0, 1, present=True, ran=True) == "fail"
    assert check_status(None, 0, present=True, ran=True) == "missing"
    assert check_status(None, 0, present=False, ran=False) == "not run"


def test_real_projection_parity():
    try:
        bundle = Bundle()
    except ValueError:
        pytest.skip("Run in the installed public projection for data parity")
    attempts = measured_attempts(bundle)
    assert len(attempts) == len({r["run_id"] for r in attempts}) == 60
    table = comparison(bundle).set_index("model")
    mini = table.loc["minimax/minimax-m2.5"]
    glm = table.loc["z-ai/glm-5"]
    assert (mini["launches"], mini["accepted"], mini["interruptions"]) == (18, 2, 1)
    assert (glm["launches"], glm["accepted"], glm["interruptions"]) == (18, 11, 2)
    assert mini["known_spend_usd"] == pytest.approx(0.54352452, abs=1e-10)
    assert glm["known_spend_usd"] == pytest.approx(1.37464460, abs=1e-10)
    assert table["complete_spend_usd"].isna().all()
    assert table["cost_per_accepted_patch_usd"].isna().all()
    assert comparison(bundle, "held_out_two")["launches"].sum() == 4
    assert bundle.read("study.json")["planned_development"]["status"] == "not run"


def test_duplicate_identities_fail_early(tmp_path):
    bundle = make_bundle(
        tmp_path / "duplicates",
        {"attempts.json": [{"run_id": "same"}, {"run_id": "same"}]},
    )
    with pytest.raises(ValueError, match="Duplicate public run identities"):
        measured_attempts(bundle)


def test_demo_deterministic_and_local_links(tmp_path):
    from html.parser import HTMLParser

    from public_replay.demo import build_demo

    try:
        bundle = Bundle()
    except ValueError:
        pytest.skip("Requires reviewed public data")
    first, second = tmp_path / "first", tmp_path / "second"
    build_demo(bundle.root, first)
    build_demo(bundle.root, second)

    def files(root):
        return {
            p.relative_to(root).as_posix(): digest(p.read_bytes())
            for p in root.rglob("*")
            if p.is_file()
        }

    assert files(first) == files(second)
    assert (first / "demo.ipynb").stat().st_size < 2_000_000

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in {"href", "src"}:
                    assert "://" not in value
                    assert safe_path(first, value).is_file(), value

    Links().feed((first / "demo.html").read_text())


def test_public_help_has_no_paid_execute(monkeypatch, capsys):
    import sys

    from public_replay.__main__ import main

    monkeypatch.setattr(sys, "argv", ["experiments", "--help"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "replay" in help_text and "prepare" in help_text
    assert "execute" not in help_text.lower()


def test_doctor_without_resources_is_honest(tmp_path):
    from public_replay.prepare import doctor

    bundle = make_bundle(tmp_path / "doctor")
    result = doctor(bundle, tmp_path / "absent")
    assert result["model_requests"] == 0
    assert result["resolved_preparation"].startswith("not prepared")
    assert result["private_verifier"].startswith("unavailable")
    assert result["paid_execution"].startswith("not exposed")
