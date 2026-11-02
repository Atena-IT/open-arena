# Releasing

The `openarena` package is published by one workflow,
`.github/workflows/release.yml`, with Trusted Publishing: PyPI and TestPyPI
trust that workflow in this repository, so no token is stored anywhere. It
builds the sdist and the wheel, checks them, publishes them, and then installs
the exact version it published on a fresh runner, runs `openarena --help`, and
checks the workspace `openarena init` writes.

For now every version goes to TestPyPI: publishing to PyPI is
[off](#turning-on-pypi) until the repository says otherwise.

| You do | Version | Goes to |
|---|---|---|
| Actions → Release → Run workflow | pyproject's, plus `.devN` | TestPyPI |
| Publish a GitHub pre-release `vX.Y.ZrcN` | `X.Y.ZrcN` | TestPyPI |
| Publish a GitHub release `vX.Y.Z` | `X.Y.Z` | TestPyPI, PyPI once it is on |

## Try a build

On GitHub, **Actions → Release → Run workflow**, on `main`. The run builds
pyproject's version with `.devN` added, N being the run's number (`0.2.0.dev7`),
so every run is a new version, and publishes it to
[TestPyPI](https://test.pypi.org/project/openarena/). The `verify` job then
downloads that wheel from TestPyPI and installs it with its dependencies from
PyPI. Its summary shows the version and the command it installed with.

Nothing here touches PyPI: try as often as you like.

## Release

1. Set the version in `pyproject.toml` (`uv version 0.3.0`), merge it into `main`.
2. On GitHub, draft a release with a new tag `v0.3.0` on `main`, the version
   with a `v` in front. Leave **Set as a pre-release** unticked.
3. Publish it. The workflow builds the tag, refuses it if the tag does not
   match pyproject's version, publishes it, and `verify` installs
   `openarena==0.3.0` from where it went.

A pre-release goes the same way, and always to TestPyPI: the version in
pyproject is `0.3.0rc1`, the tag `v0.3.0rc1`, and **Set as a pre-release** is
ticked.

!!! warning "A version is published once"
    Neither PyPI nor TestPyPI ever takes the same version twice, even after
    it is deleted. A release whose publish failed halfway, or that shipped a
    mistake, is fixed with a new version (`0.3.1`), not by publishing the tag
    again. A release that went to TestPyPI while PyPI was off can still go to
    PyPI later under the same version: the two indexes are separate.

## Turning on PyPI

From then on a release that is not a pre-release goes to
[PyPI](https://pypi.org/project/openarena/) instead of TestPyPI, and `verify`
runs `pip install openarena==X.Y.Z`.

1. On PyPI, a pending publisher for project `openarena`: owner `Atena-IT`,
   repository `open-arena`, workflow `release.yml`, environment `pypi`.
2. In the repository's Settings → Environments, an environment `pypi` that
   only tags `v*` may deploy to, with no secrets.
3. In Settings → Secrets and variables → Actions → Variables, a repository
   variable `PUBLISH_PYPI` set to `true`.
