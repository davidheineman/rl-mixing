#!/usr/bin/env python3
"""Patch open-instruct to support:
1. Comma-separated multi-mapping in --remap_verifier
2. LeanVerifier for Lean 4 proof compilation

Run from the open-instruct repo root after cloning:
    python /path/to/patch_open_instruct.py
"""
import pathlib
import shutil
import sys

OI_ROOT = pathlib.Path(".")
GTU = OI_ROOT / "open_instruct" / "ground_truth_utils.py"


def patch_multi_remap():
    """Upgrade remap_verifier from single old=new to comma-separated old1=new1,old2=new2."""
    t = GTU.read_text()
    old = (
        '        remap = streaming_config.remap_verifier.split("=")\n'
        '        assert len(remap) == 2, "Remap must be in the format old_name=new_name"\n'
        '        old_name, new_name = remap\n'
        '        # map so that the old name calls the new verifier\n'
        '        assert new_name.lower() in verifiers, f"{new_name} not found in verifiers during remapping"\n'
        '        verifiers[old_name.lower()] = verifiers[new_name.lower()]'
    )
    new = (
        '        for _m in streaming_config.remap_verifier.split(","):\n'
        '            _m = _m.strip()\n'
        '            if not _m:\n'
        '                continue\n'
        '            remap = _m.split("=")\n'
        '            assert len(remap) == 2, f"Remap format: old=new, got: {_m}"\n'
        '            old_name, new_name = remap\n'
        '            assert new_name.lower() in verifiers, f"{new_name} not found in verifiers during remapping"\n'
        '            verifiers[old_name.lower()] = verifiers[new_name.lower()]'
    )
    if old in t:
        GTU.write_text(t.replace(old, new, 1))
        print("[patch] remap_verifier: upgraded to multi-mapping")
    elif "for _m in" in t:
        print("[patch] remap_verifier: already patched")
    else:
        print("[patch] remap_verifier: WARNING - source changed, could not patch", file=sys.stderr)
        return False
    return True


def install_lean_verifier():
    """Copy lean_verifier.py into open_instruct/ and register its import."""
    src = pathlib.Path(__file__).parent / "lean_verifier.py"
    dst = OI_ROOT / "open_instruct" / "lean_verifier.py"
    if not src.exists():
        print(f"[patch] lean_verifier: source not found at {src}", file=sys.stderr)
        return False
    shutil.copy2(src, dst)
    print(f"[patch] lean_verifier: copied to {dst}")

    t = GTU.read_text()
    import_line = "from open_instruct.lean_verifier import LeanVerifier  # noqa: F401"
    if import_line not in t:
        # Add import after the last existing import block
        marker = "from open_instruct.rubrics.run_utils import"
        if marker in t:
            t = t.replace(marker, f"{import_line}\n{marker}", 1)
        else:
            # fallback: add at the top of the file after the docstring
            t = t.replace('"""', '"""\n' + import_line, 1)
        GTU.write_text(t)
        print("[patch] lean_verifier: registered import in ground_truth_utils.py")
    else:
        print("[patch] lean_verifier: import already present")
    return True


if __name__ == "__main__":
    if not GTU.exists():
        print(f"ERROR: {GTU} not found (run from open-instruct root)", file=sys.stderr)
        sys.exit(1)

    ok = True
    ok &= patch_multi_remap()
    ok &= install_lean_verifier()

    if not ok:
        print("\nSome patches failed!", file=sys.stderr)
        sys.exit(1)
    print("\nAll patches applied successfully.")
