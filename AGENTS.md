# Farpoint Agent Rules

Follow `CONTRIBUTING.md` and `docs/development-workflow.md` for every change.

- Never commit or push directly to `main`.
- Create a feature branch and a Draft PR.
- Never merge a PR, enable auto-merge, create a production release tag, or
  publish to Hugging Face without an explicit owner request after review.
- Treat `configs/datasets/*.toml` as dataset release-version sources of truth.
- Keep Hugging Face Dataset Cards outside Git; review and edit them directly on
  the Hub only during an owner-approved dataset publication.
- Keep pilot/debug artifacts separate from formal benchmarks and releases.
- Report test, pilot, validation, and publishing evidence separately.

## Architecture

- Keep experiment-specific values in versioned configuration or immutable
  evidence, not reusable runtime modules.
- New variation axes must not require a new collection engine, manifest type,
  selector, or exporter.
- Prefer generic entity paths, validators, samplers, split policies, and
  selection constraints over mass-, yaw-, shape-, or release-specific code.
- Preserve backward reading compatibility for published schemas and manifests.
- Do not change frozen plan hashes, release evidence, or immutable dataset tags.
