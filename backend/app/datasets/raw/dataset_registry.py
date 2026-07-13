# medical-triage-agent-ai-poc/backend/app/datasets/raw/dataset_registry.py

"""
Registry central des datasets médicaux.
"""

DATASET_REGISTRY = {
    "mediqa": {
        "hf_path": "ANR-MALADES/MediQAl",
        "subset": "oeq",
        "description": "French open-ended medical QA dataset",
    },
    "mediqa_mcqu": {
        "hf_path": "ANR-MALADES/MediQAl",
        "subset": "mcqu",
        "description": "French single-answer medical MCQU - QA dataset",
    },
    "mediqa_mcqm": {
        "hf_path": "ANR-MALADES/MediQAl",
        "subset": "mcqm",
        "description": "French multi-answer medical MCQM - QA dataset",
    },
    "medquad": {
        "hf_path": "lavita/MedQuAD",
        "subset": None,
        "description": "English medical QA dataset",
    },
    "French_Med_MCQA": {
        "hf_path": "uy-rrodriguez/FrenchMedMCQA-extended",
        "subset": None,
        "description": "French medical QA dataset",
    },
    "ultramedical_preference": {
        "hf_path": "TsinghuaC3I/UltraMedical-Preference",
        "subset": None,
        "description": "Medical preference dataset for DPO",
    },
}
