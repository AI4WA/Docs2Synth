# PyPI Publishing Guide

This document explains how to publish Docs2Synth to PyPI using GitHub Actions.

## Overview

The package uses an automated GitHub Actions workflow (`.github/workflows/publish.yml`) to build and publish to PyPI.

## Setup Options

### Option 1: Trusted Publishing (Recommended)

Trusted Publishing is the modern, more secure approach that doesn't require API tokens.

**Steps:**

1. **Configure PyPI Trusted Publisher:**
   - Go to https://pypi.org/manage/project/docs2synth/settings/publishing/ (after initial upload)
   - Or for new projects: https://pypi.org/manage/account/publishing/
   - Click "Add a new publisher"
   - Fill in:
     - **Owner**: `AI4WA` (your GitHub org/username)
     - **Repository**: `Docs2Synth`
     - **Workflow**: `publish.yml`
     - **Environment**: `pypi`

2. **Configure TestPyPI (optional but recommended):**
   - Go to https://test.pypi.org/manage/account/publishing/
   - Add publisher with environment name: `testpypi`

3. **Done!** The workflow is already configured for trusted publishing.

### Option 2: API Token Method

If you prefer using API tokens:

**Steps:**

1. **Generate PyPI API Token:**
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Name: `GitHub Actions - Docs2Synth`
   - Scope: "Entire account" or specific to `docs2synth` project
   - Copy the token (starts with `pypi-...`)

2. **Generate TestPyPI Token (optional):**
   - Go to https://test.pypi.org/manage/account/token/
   - Create token same as above

3. **Add Secrets to GitHub:**
   - Go to https://github.com/AI4WA/Docs2Synth/settings/secrets/actions
   - Click "New repository secret"
   - Add:
     - Name: `PYPI_API_TOKEN`
     - Value: Your PyPI token
   - Add another:
     - Name: `TEST_PYPI_API_TOKEN`
     - Value: Your TestPyPI token

4. **Update Workflow:**
   - Edit `.github/workflows/publish.yml`
   - Uncomment the `password` lines and comment out `id-token: write`
   - See comments in the workflow file

## Publishing

### Method 1: Tag-Based Release (Recommended)

When you push a version tag, it automatically publishes:

```bash
# Update version in pyproject.toml first
# Then create and push a tag
git tag v0.1.0
git push origin v0.1.0
```

The workflow will:
1. Build the package
2. Publish to TestPyPI
3. If TestPyPI succeeds, publish to PyPI

### Method 2: Manual Trigger

Publish manually from GitHub Actions UI:

1. Go to https://github.com/AI4WA/Docs2Synth/actions/workflows/publish.yml
2. Click "Run workflow"
3. Choose where to publish:
   - `testpypi` - Test only
   - `pypi` - Production only (skips test)
   - `both` - Test then production

## First-Time Publishing

For the very first publish to PyPI (when the project doesn't exist yet):

1. **Manual upload first time:**
   ```bash
   # Build locally
   python -m build

   # Upload to PyPI
   twine upload dist/*
   ```

2. **Then set up Trusted Publishing** (see Option 1 above)

3. **Future releases** will be automated via GitHub Actions

## Testing the Package

After publishing to TestPyPI, test installation:

```bash
# CPU version
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple docs2synth[cpu]

# Check it works
docs2synth --version
```

## Version Management

Update version in `pyproject.toml`:

```toml
[project]
name = "docs2synth"
version = "0.1.1"  # Update this
```

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (1.0.0): Breaking changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes

## Troubleshooting

**"File already exists" error:**
- You're trying to upload the same version twice
- Update version in `pyproject.toml`

**"Invalid authentication" error:**
- Check your API token is correctly set in GitHub Secrets
- Token should start with `pypi-`
- Make sure secret name matches workflow file

**"Project name not found" (Trusted Publishing):**
- Must do first upload manually before Trusted Publishing works
- Or pre-register project name at https://pypi.org/manage/account/publishing/

## Resources

- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for PyPI](https://github.com/marketplace/actions/pypi-publish)
