# Jailbreaking Multimodal Large Language Models using Multi-Clip Video
MCV-SafetyBench is a video-based safety benchmark for evaluating multimodal large language models under multi-clip jailbreak attacks.
[[Paper]](https://aclanthology.org/2026.acl-long.1186/)  [[Dataset]](https://huggingface.co/datasets/Choongwon/MCV_SafetyBench)
### Overview of the Multi-Clip Video SafetyBench
![Figure](Fig/Fig_2.png)

## Repository Structure
### 1. Attack
```text
Attack/
├── Explicit/
│   └── Changed_question/
└── Implicit/
```
#### Description
- `Attack/Explicit/` contains the code used to perform explicit jailbreak attacks on multimodal large language models (MLLMs).
- `Attack/Implicit/` contains the code used to perform implicit jailbreak attacks on multimodal large language models (MLLMs).
- `Attack/Explicit/Changed_question/` contains the prompts used for the explicit attack setting.

### 2. Evaluation
```text
Evaluation/
├── policy.txt
├── score.txt
└── score_judgement.py
```

#### Description
- `Evaluation/policy.txt` contains the safety policy used for response evaluation.
- `Evaluation/score.txt` contains the 1–5 scoring rubric for judging model responses.
- `Evaluation/score_judgement.py` contains the GPT-4o-mini-based scoring script. We treat an attack as successful only when the response receives `Score 5`.

### 3. Prompt Construction
```text
Prompt_construction/
├── semantic_extraction.py
└── semantic_reconstruction.py
```

#### Description
- `Prompt_construction/semantic_extraction.py` contains the code used to extract semantic phrases from harmful queries, including subject, object, action, and mood phrase.
- `Prompt_construction/semantic_reconstruction.py` contains the code used to reconstruct the extracted phrases into a complete sentence.
- These codes were used to construct MCV-SafetyBench by using GPT-4o.
