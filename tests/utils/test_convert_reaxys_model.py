import csv
import gzip
import json
import zipfile

import pytest

from aizynthfinder.tools.convert_reaxys_model import (
    _convert_templates,
    _parenthesize_template,
    _sha256,
)


def _write_archive(path, indices=(0, 1)):
    with zipfile.ZipFile(path, "w") as archive:
        rows = []
        for index in indices:
            rows.append(
                json.dumps(
                    {
                        "index": index,
                        "_id": f"template-{index}",
                        "count": index + 2,
                        "dimer_only": False,
                        "intra_only": True,
                        "necessary_reagent": "",
                        "reaction_smarts": f"[C:{index + 1}]>>[O:{index + 1}]",
                        "references": ["omitted-from-output"],
                        "template_set": "reaxys",
                    }
                )
            )
        archive.writestr("templates.jsonl", "\n".join(rows) + "\n")


def test_parenthesize_template():
    assert _parenthesize_template("[C:1]>>[O:1]") == "([C:1])>>([O:1])"

    with pytest.raises(ValueError, match="one reaction arrow"):
        _parenthesize_template("[C:1]")


def test_convert_templates_is_deterministic_and_omits_references(tmp_path):
    archive_path = tmp_path / "test.mar"
    first_output = tmp_path / "first.csv.gz"
    second_output = tmp_path / "second.csv.gz"
    _write_archive(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        assert _convert_templates(archive, first_output, expected_count=2) == 2
    with zipfile.ZipFile(archive_path) as archive:
        assert _convert_templates(archive, second_output, expected_count=2) == 2

    assert _sha256(first_output) == _sha256(second_output)
    with gzip.open(first_output, "rt", encoding="utf-8", newline="") as fileobj:
        rows = list(csv.DictReader(fileobj, delimiter="\t"))

    assert rows == [
        {
            "template_code": "0",
            "retro_template": "([C:1])>>([O:1])",
            "count": "2",
            "dimer_only": "False",
            "intra_only": "True",
            "necessary_reagent": "",
            "template_set": "reaxys",
            "_id": "template-0",
        },
        {
            "template_code": "1",
            "retro_template": "([C:2])>>([O:2])",
            "count": "3",
            "dimer_only": "False",
            "intra_only": "True",
            "necessary_reagent": "",
            "template_set": "reaxys",
            "_id": "template-1",
        },
    ]
    assert "references" not in rows[0]


def test_convert_templates_rejects_non_contiguous_indices(tmp_path):
    archive_path = tmp_path / "test.mar"
    _write_archive(archive_path, indices=(0, 3))

    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="indices must be contiguous"):
            _convert_templates(archive, tmp_path / "templates.csv.gz", expected_count=2)
