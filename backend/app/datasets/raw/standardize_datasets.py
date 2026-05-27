# medical-triage-agent-ai-poc/backend/app/datasets/raw/standardize_datasets.py

"""
Pipeline standardisation datasets médicaux.
"""

from pathlib import Path
import json

INPUT_DIR = Path("backend/app/datasets/raw/data")
OUTPUT_DIR = Path(
    "backend/app/datasets/raw/standardized"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def standardize_record(record, source_name):

    standardized = {
        "id": str(record.get("id", "")),
        "instruction": (
            record.get("question")
            or record.get("instruction")
            or ""
        ),
        "response": (
            record.get("answer")
            or record.get("response")
            or ""
        ),
        "source": source_name,
        "language": "fr",
        "metadata": {
            "original_fields": list(record.keys())
        },
    }

    return standardized


def process_file(file_path):

    output_path = OUTPUT_DIR / file_path.name

    with open(file_path, "r", encoding="utf-8") as infile:
        with open(output_path, "w", encoding="utf-8") as outfile:

            for line in infile:

                record = json.loads(line)

                standardized = standardize_record(
                    record,
                    file_path.stem,
                )

                outfile.write(
                    json.dumps(
                        standardized,
                        ensure_ascii=False,
                    ) + "\n"
                )

    print(f"Standardized -> {output_path}")


def main():

    files = INPUT_DIR.glob("*.jsonl")

    for file_path in files:
        process_file(file_path)


if __name__ == "__main__":
    main()
