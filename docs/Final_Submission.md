# Final Submission Checklist (DSCI351)

This file reflects the current submission plan: the professor said a separate final report is no longer required, so the repo, final presentation, reproducible code, dataset citation, and artifacts are the submission package.

## Required Artifacts

| Artifact | Status | Location |
|----------|--------|----------|
| Project repository | Ready | Root README + `src/`, `scripts/`, `notebooks/`, `artifacts/` |
| Final presentation | Ready | `docs/Final Presentation.pptx` |
| Canvas project prompt | Stored for reference | `docs/canvas/DSCI351-project.pdf` |
| Dataset citation/link | Ready | Root README, `data/README.md`, slide 6 |
| Reproducible metrics | Ready | `artifacts/runs/n*/results/model_comparison.csv` |
| Figures/results for grading | Ready | `artifacts/runs/n*/figures/`, `artifacts/comparisons/` |

## Rubric Alignment

| Rubric area | Where it is covered |
|-------------|---------------------|
| Dataset description | README dataset section, `data/README.md`, slide 6 |
| Recommender system type | README overview, `docs/How_It_Works.md`, slides 3-4 |
| Methods/models | `src/models/`, `docs/Pipeline.md`, slides 8-12 |
| Interpretation of recommendations | README strategy examples, slides 17-18 |
| Evaluation | `src/evaluation.py`, `artifacts/runs/n*/results/`, slides 13-16 |
| Limitations/future work | README limitations references, slide 19 |

## Presentation Coverage

The final deck includes:

- Title/team and problem statement
- Users/items/implicit feedback/top-N recommendation definition
- Dataset source, scale, categories, and engineered strategy label
- Engagement metric and pseudo-rating definition
- Pipeline overview
- Global/category baselines, user-based CF, content-based, and hybrid models
- Time-based split and Precision@5, Recall@5, NDCG@5, Hit Rate@5
- 10k and 50k result slides plus NDCG interpretation
- Two influencer case studies
- Limitations and next steps

## Repository QA

- [x] `requirements.txt` lists required dependencies.
- [x] README explains setup, local synthetic reproduction, and real-data/Colab workflow.
- [x] Code produces evaluation metrics and figures from `scripts/run_pipeline.py`.
- [x] Precomputed metrics and figures are committed for graders.
- [x] Large raw/processed files are gitignored.
- [x] No personal Drive paths are required for local reproduction.
- [x] Hybrid weight tuning uses a validation split from the training period, not final test labels.

## Dataset Links

- Official dataset page: https://sites.google.com/site/sbkimcv/dataset/instagram-influencer-dataset
- Google Drive mirror: https://drive.google.com/drive/folders/1ISiSH4-aM6kP_0lKQYejpk1sa6Jei-7e

## Before Uploading

1. Commit `docs/canvas/DSCI351-project.pdf` if the Canvas prompt must live in the submitted repo.
2. Submit the repo link and `docs/Final Presentation.pptx`.
3. Mention that a separate final report was waived by the professor if Canvas still shows an old report placeholder.
