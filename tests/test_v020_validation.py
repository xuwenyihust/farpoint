from farpoint import v020_validation


def _report(monkeypatch, lerobot_validation):
    monkeypatch.setattr(
        v020_validation, "validate_v020_selection", lambda plans, selection: []
    )
    return v020_validation.build_v020_candidate_validation(
        [{"plan_sha256": "a" * 64, "campaign_sha256": "b" * 64}],
        {"episodes": []},
        lerobot_validation=lerobot_validation,
    )


def test_candidate_validation_accepts_native_lerobot_report(monkeypatch):
    report = _report(monkeypatch, {"valid": True, "errors": []})
    assert report["status"] == "PASS"
    assert report["errors"] == []


def test_candidate_validation_keeps_status_pass_compatibility(monkeypatch):
    report = _report(monkeypatch, {"status": "PASS", "errors": []})
    assert report["status"] == "PASS"


def test_candidate_validation_rejects_invalid_native_lerobot_report(monkeypatch):
    report = _report(monkeypatch, {"valid": False, "errors": ["broken video"]})
    assert report["status"] == "FAIL"
    assert report["errors"] == ["lerobot_validation_not_pass"]
