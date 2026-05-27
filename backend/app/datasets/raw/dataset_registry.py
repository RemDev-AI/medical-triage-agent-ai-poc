# medical-triage-agent-ai-poc/backend/app/datasets/raw/dataset_registry.py

"""
Registry central des datasets médicaux.
"""

DATASET_REGISTRY = {
    "mediqa": {
        "hf_path": "bigbio/mediqa",
        "subset": None,
        "description": "Medical QA benchmark dataset",
    },
    "frenchmedmcqa": {
        "hf_path": "mlalouani/frenchmedmcqa",
        "subset": None,
        "description": "French medical multiple choice QA",
    },
    "medquad": {
        "hf_path": "lavita/MedQuAD",
        "subset": None,
        "description": "Medical question answering dataset",
    },
    "ultramedical_preference": {
        "hf_path": "KaiLv/UltraMedical",
        "subset": None,
        "description": "Medical preference alignment dataset",
    },
}
