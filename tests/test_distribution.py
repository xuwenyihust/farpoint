import zipfile

from scripts.check_distribution import REQUIRED_MEMBERS, check_distribution


def test_distribution_check_requires_runtime_schemas(tmp_path):
    wheel = tmp_path / "farpoint-1.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for member in sorted(REQUIRED_MEMBERS):
            archive.writestr(member, "{}")
    assert check_distribution(tmp_path) == []


def test_distribution_check_reports_missing_schema(tmp_path):
    wheel = tmp_path / "farpoint-1.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("farpoint/contracts.py", "")
    errors = check_distribution(tmp_path)
    assert any("farpoint_dataset_v2.schema.json" in error for error in errors)
