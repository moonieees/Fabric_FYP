import re
import shutil
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Matches D1_C1S1.jpg, D14_PIS8.jpg, CCS1.jpg, or D0_CPS2.jpg
OLD_PATTERN = re.compile(
    r"^(?:D(\d+)_)?(CC|CP|CI|PI|C[1-3]|P[1-3])S(\d+)$",
    re.IGNORECASE
)

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def organize_and_rename():
    if not STAGING_DIR.exists():
        print(f"Staging folder not found: {STAGING_DIR}")
        return

    copied_count = 0
    unmatched_files = []

    for img_path in sorted(STAGING_DIR.rglob("*")):
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        match = OLD_PATTERN.match(img_path.stem)
        if not match:
            unmatched_files.append(img_path.name)
            continue

        day_str, group_code, sample_str = match.groups()
        group_code = group_code.upper()
        day = 0 if group_code in {"CC", "CP"} else int(day_str)
        sample_num = int(sample_str)

        # Build new filename: e.g., C1_D1_1.jpg
        new_filename = f"{group_code}_D{day}_{sample_num}{img_path.suffix.lower()}"
        target_folder = RAW_DIR / group_code
        target_folder.mkdir(parents=True, exist_ok=True)

        target_path = target_folder / new_filename
        shutil.copy2(img_path, target_path)
        copied_count += 1

    print(f"Done! Successfully renamed and sorted {copied_count} images into: {RAW_DIR}")
    if unmatched_files:
        print(f"\nWarning - {len(unmatched_files)} files did not match the naming pattern:")
        for name in unmatched_files:
            print("  -", name)

if __name__ == "__main__":
    organize_and_rename()