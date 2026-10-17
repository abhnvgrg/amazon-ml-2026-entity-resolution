"""Build the competition submission zips (D-013).

submissions/{tag}_code.zip is a code-only snapshot; submissions/{tag}/{team}_submission.zip is
the final package with output/, code/business_entity_resolution/ and the methodology document.
Run: python src/make_package.py sub-04b --team <team_name>
"""

from __future__ import annotations

import argparse
import zipfile

from common import OUT, ROOT


def code_files() -> list:
    src = ROOT / "code" / "business_entity_resolution"
    files = [src / "README.md", src / "requirements.txt"]
    files += sorted(p for p in (src / "src").rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and p.suffix != ".pyc")
    missing = [p for p in files if not p.exists()]
    assert not missing, f"missing: {missing}"
    return files


def write_zip(out, entries: list) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p, arc in entries:
            z.write(p, arc)
    print(f"{out}: {len(entries)} files, {out.stat().st_size:,} bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="e.g. sub-04")
    ap.add_argument("--team", required=True, help="team name for <team>_submission.zip")
    a = ap.parse_args()
    code = [(p, p.relative_to(ROOT).as_posix()) for p in code_files()]
    write_zip(ROOT / "submissions" / f"{a.tag}_code.zip",
              [(p, arc.removeprefix("code/")) for p, arc in code])
    outs = [OUT / "matching_results.tsv", OUT / "candidate_pairs.tsv"]
    doc = ROOT / "Documentation_template.md"
    for p in outs + [doc]:
        assert p.exists(), f"missing: {p}"
    final = [(p, f"output/{p.name}") for p in outs] + code + [(doc, doc.name)]
    out = ROOT / "submissions" / a.tag / f"{a.team}_submission.zip"
    write_zip(out, final)
    with zipfile.ZipFile(out) as z:
        for i in z.infolist():
            print(f"  {i.filename}  ({i.file_size:,} B)")


if __name__ == "__main__":
    main()
