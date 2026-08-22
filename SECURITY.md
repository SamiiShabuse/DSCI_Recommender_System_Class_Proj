# Security Policy

## Scope

This repository is primarily a research and coursework project. The main security and privacy risks are accidental commits of raw data, credentials, or local paths.

## Do Not Commit

- API keys, tokens, or `.env` files
- Raw Instagram metadata dumps
- Large extracted datasets or private data exports
- Local absolute paths from a personal machine
- Generated parquet files unless they are intentionally sanitized and documented

## Reporting a Concern

If you find sensitive data or a security issue, report it privately to the repository owner instead of opening a public issue. Include the file path, the type of exposure, and whether the issue appears in the latest commit or repository history.