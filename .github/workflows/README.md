# OGX CI

This is a fork of OGX. CI is intentionally minimal and scoped to fast, local-style
checks: unit tests, lint/type-checking, and build/compile sanity checks. Integration
tests, release/publishing automation, and external-provider workflows from upstream
have been removed.

| Name | File | Purpose |
| ---- | ---- | ------- |
| Build and Push on oracle-dev | [build-push-oracle-dev.yml](build-push-oracle-dev.yml) | Build image and push to OCIR on `oracle-dev` push (versioned tag) |
| Build and Push on PR | [build-push-pr.yml](build-push-pr.yml) | Build image and push to OCIR on PR (short-SHA tag) |
| CodeQL Workflow Security Scan | [codeql.yml](codeql.yml) | CodeQL scan of `.github/**` on PR |
| Pre-commit | [pre-commit.yml](pre-commit.yml) | Run pre-commit checks (ruff, mypy, hooks) |
| Test OGX Build | [providers-build.yml](providers-build.yml) | Build distros (venv + container) — compile sanity |
| Test ogx stack list-deps | [providers-list-deps.yml](providers-list-deps.yml) | Resolve and install distro dependencies |
| UI Tests | [ui-unit-tests.yml](ui-unit-tests.yml) | UI lint, format check, and unit tests |
| Unit Tests | [unit-tests.yml](unit-tests.yml) | Python unit test suite (3.12, 3.13) |
