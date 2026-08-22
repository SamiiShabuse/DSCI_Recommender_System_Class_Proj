# Contributing

Thanks for improving the Creator Intelligence Recommender System. This project is a course-quality recommender systems pipeline, so contributions should keep the repository reproducible, readable, and honest about the data requirements.

## Good Contributions

- Clarify setup or reproduction steps.
- Improve synthetic pipeline reliability.
- Add focused tests for preprocessing, model ranking, or evaluation metrics.
- Improve figures, result summaries, or documentation links.
- Add experiments only when the metric, data split, and run configuration are documented.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the synthetic pipeline when validating code changes:

```bash
python scripts/run_pipeline.py --synthetic --target-posts 1000
python -m unittest discover -s tests
```

## Data Guidelines

Do not commit raw Instagram metadata, extracted full datasets, local parquet outputs, or personal credentials. Keep large or private data in local storage, Google Drive, or another controlled location. Public commits should include code, documentation, aggregate results, and reproducible instructions.

## Pull Request Checklist

- The change has a clear purpose.
- Setup or run instructions were updated if behavior changed.
- Tests or a documented smoke run were completed.
- Generated artifacts are intentional and small enough for GitHub.
- No raw dataset files, credentials, or local machine paths are included.