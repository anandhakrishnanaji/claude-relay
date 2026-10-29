# Releasing claude-relay

A release publishes three things from one git tag:

| What | Where |
|---|---|
| Python package | [PyPI: `claude-relay-server`](https://pypi.org/project/claude-relay-server/) |
| Docker image (amd64 + arm64) | `ghcr.io/anandhakrishnanaji/claude-relay` tagged `X.Y.Z`, `X.Y` and `latest` |
| GitHub release | wheel + sdist attached, notes generated from commits |

`.github/workflows/release.yml` does all of it when a `vX.Y.Z` tag is pushed.

## Names

The PyPI name `claude-relay` belongs to an unrelated project, so the names differ by context:

| Context | Name |
|---|---|
| PyPI distribution | `claude-relay-server` |
| Python import package | `claude_relay` |
| Commands | `claude-relay` (primary) and `claude-relay-server` (so `uvx claude-relay-server` works) |
| Docker image | `ghcr.io/anandhakrishnanaji/claude-relay` |
| systemd unit, `/health` service field | `claude-wrapper` (kept for existing installs) |

## One-time setup

Do this once per project (and again if the repo or workflow file is renamed).

1. **PyPI trusted publisher.** On PyPI, open `claude-relay-server` → Settings → Publishing → Add a GitHub publisher:
   owner `anandhakrishnanaji`, repository `claude-relay`, workflow `release.yml`, environment `pypi`.
   The workflow then uploads with a short-lived OIDC token, so no PyPI API token is stored anywhere.
2. **GitHub environment.** The `pypi` environment is created automatically on the first run. Optionally add yourself as a required reviewer under Settings → Environments → `pypi` to approve each upload by hand.
3. **GHCR visibility.** The image inherits the public repository's visibility. If a pull ever asks for a login, set the `claude-relay` package to public under your GitHub profile → Packages.

## Cutting a release

1. **Bump the version** in both places (they must match):
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `claude_relay/__init__.py` → `__version__ = "X.Y.Z"`

   Use a major bump for breaking changes (module path, defaults, removed flags or env vars), minor for features, patch for fixes. Update the "Current Version" notes in `README.md`.

2. **Check locally** with the same tools CI uses:
   ```bash
   uvx black==24.10.0 --check claude_relay tests   # CI pins black 24.10; newer versions format differently
   uvx bandit -r claude_relay/ -ll
   .venv/bin/python -m pytest tests -q
   uv build && uvx --from ./dist/claude_relay_server-X.Y.Z-py3-none-any.whl claude-relay --version
   ```
   Three tests in `tests/test_auth_unit.py` fail locally if the repo has a `.env` setting `CLAUDE_AUTH_METHOD`; they pass in CI, which has no `.env`.

3. **Commit, push, and wait for CI to pass on `main`.** Tag only a green commit: the release workflow runs from the tagged commit, so a broken commit means a broken release.

4. **Tag and push the tag:**
   ```bash
   git tag -a vX.Y.Z -m "claude-relay-server X.Y.Z"
   git push origin vX.Y.Z
   ```

5. **Watch the run:** `gh run watch -R anandhakrishnanaji/claude-relay` (or the Actions tab).

## What the workflow does

| Job | Steps |
|---|---|
| `build` | Checks the tag matches `pyproject.toml`; checks whether the version is already on PyPI; runs the tests; builds the wheel and sdist; installs the wheel and runs `claude-relay --version` |
| `pypi` | Uploads to PyPI via trusted publishing. **Skipped** if the version is already on PyPI (for example, uploaded by hand) |
| `docker` | Builds `Dockerfile` for amd64 and arm64 and pushes to GHCR |
| `github-release` | Creates the GitHub release with the built files. Runs even when `pypi` was skipped |

## Checking a release

```bash
uvx --refresh claude-relay-server --version
docker pull ghcr.io/anandhakrishnanaji/claude-relay:X.Y.Z
gh release view vX.Y.Z -R anandhakrishnanaji/claude-relay
```

## When something goes wrong

**The tag doesn't match the version.** The `build` job fails at its first check. Delete the tag (`git push origin :refs/tags/vX.Y.Z`), fix the version, and tag again.

**The workflow fails before anything is published.** Fix it on `main`, then move the tag to the fixed commit. Re-running the old run won't help, because it uses the workflow file from the tagged commit:
```bash
git tag -f vX.Y.Z <fixed-commit>
git push -f origin vX.Y.Z
```
Only move a tag if nothing was published from it yet.

**PyPI uploaded but a later job failed.** A PyPI version can never be re-uploaded, even after deleting it. Fix the problem and move the tag as above: the `build` job sees the version is already on PyPI, skips the upload, and the Docker and GitHub release jobs run normally. If the package itself was broken, release `X.Y.Z+1` instead.

**The `pypi` job fails with an OIDC or "invalid-publisher" error.** The trusted publisher is missing or its fields don't match (owner, repository, workflow file name `release.yml`, environment `pypi`). Fix it on PyPI and re-run the failed job.

**Publishing by hand** (only if the workflow can't be used). Create a project-scoped API token on PyPI, then:
```bash
uv build
UV_PUBLISH_TOKEN=pypi-... uv publish dist/*
```
Revoke the token afterwards. Pushing the tag later still produces the Docker image and GitHub release, because the PyPI upload is skipped.

## Deployments using the old layout

Before 3.0.0 the package directory was `src/` and the server ran as `uvicorn src.main:app`. Existing installs need:

- **systemd service:** rerun `./scripts/install-service.sh`. It keeps `.env` and switches the unit to `.venv/bin/claude-relay`.
- **Docker built from a checkout:** copy `claude_relay/` instead of `src/` into the build context and rebuild.
