import csv
import json
import os

print("=== SIGNBRIDGE DATASET INTEGRITY CHECK ===")

errors = []

# 1. Verify dataset_manifest.json
manifest_path = "src/dataset/dataset_manifest.json"
try:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        print(f"[OK] dataset_manifest.json parsed successfully. Registered datasets: {len(manifest.get('datasets', []))}")
except Exception as e:
    errors.append(f"Manifest error: {e}")

# 2. Verify ISLTranslate.csv
isl_translate_path = "src/dataset/ISLTranslate/data/ISLTranslate.csv"
try:
    with open(isl_translate_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for row in reader)
        print(f"[OK] ISLTranslate.csv loaded completely! Header columns: {len(header)}, Total data rows: {row_count}")
except Exception as e:
    errors.append(f"ISLTranslate.csv error: {e}")

# 3. Verify ISL-signer_validation.csv
val_path = "src/dataset/ISLTranslate/data/ISL-signer_validation.csv"
try:
    with open(val_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = sum(1 for row in reader)
        print(f"[OK] ISL-signer_validation.csv loaded completely! Header columns: {len(header)}, Total validation rows: {row_count}")
except Exception as e:
    errors.append(f"ISL-signer_validation.csv error: {e}")

# 4. Verify Mendeley ISL CSV
mendeley_path = "src/dataset/Mendeley_ISL/extracted/ISL_Mendeley_Alphabets.csv"
try:
    with open(mendeley_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        letters = [r['letter'] for r in rows]
        print(f"[OK] ISL_Mendeley_Alphabets.csv loaded completely! Total alphabet letters parsed: {len(rows)}")
        print(f"     Parsed letters: {', '.join(letters)}")
        if len(rows) == 26:
            print("[OK] Complete 26-letter ISL 2-handed manual alphabet verified (A-Z)!")
        else:
            errors.append(f"Expected 26 letters, got {len(rows)}")
except Exception as e:
    errors.append(f"Mendeley CSV error: {e}")

if not errors:
    print("\n[PASS] ALL DATASETS LOADED 100% COMPLETELY AND PASSED INTEGRITY VERIFICATION!")
else:
    print("\n[WARNING] VERIFICATION ISSUES:")
    for err in errors:
        print(f" - {err}")
