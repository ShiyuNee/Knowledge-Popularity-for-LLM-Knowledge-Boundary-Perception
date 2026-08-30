# Public Release Checklist

Use this checklist before making the repository public.

## Recommended strategy: start a clean public history

Large experiment artifacts existed in the private research history. Deleting
them in a later commit does not remove their blobs from earlier commits.
Therefore, the recommended publication strategy is:

1. Keep this research repository private as the archival working copy.
2. Export the cleaned working tree to a new directory.
3. Initialize a new Git repository in that directory.
4. Publish the new repository as the public code release.

Example:

```bash
mkdir -p ../knowledge-popularity-public

rsync -av ./ ../knowledge-popularity-public/ \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude 'res/' \
  --exclude 'llm_pop_generation/' \
  --exclude 'baselines/self_consistency/' \
  --exclude 'baselines/verbalized_confidence/' \
  --exclude 'pop_generation/data/' \
  --exclude 'artifacts/' \
  --exclude 'downloads/' \
  --exclude 'models/' \
  --exclude 'checkpoints/' \
  --exclude 'paper_writing/anthology.bib' \
  --exclude '*.zip' \
  --exclude '*.tar*' \
  --exclude '*.parquet'

cd ../knowledge-popularity-public
git init
git add .
python scripts/check_public_release.py --working-tree
git commit -m "Initial public release"
```

Inspect the new repository before adding a remote:

```bash
git status --short
git ls-files
git count-objects -vH
```

Only then create and push to the public GitHub repository.

## Do not publish by only committing deletions

The following operation stops tracking a local artifact directory:

```bash
git rm -r --cached res
```

It does **not** remove `res/` from existing commits. If the current repository
history is pushed, those blobs remain downloadable.

History rewriting with `git filter-repo` is possible, but it changes commit
hashes and can disrupt collaborators. Prefer a clean public history unless
preserving the current history is essential.

## Content checks

- [ ] `README.md` describes the current paper title and method.
- [ ] The paper PDF and LaTeX source correspond to the same version.
- [ ] `CITATION.cff` or the README citation contains final author metadata.
- [ ] Anonymous submission-only material has been removed if anonymity is no
      longer required.
- [ ] The lightweight QA data may be redistributed under upstream terms.
- [ ] `docs/DATA_AND_ARTIFACTS.md` names the exact Wikipedia snapshot.
- [ ] Model identifiers and generation parameters match the paper.
- [ ] Self-consistency is consistently documented as 10 samples.
- [ ] Compact result JSON files correspond to the published tables.
- [ ] No API keys, proxy URLs, passwords, usernames, or private server paths
      remain in source or documentation.

## Automated checks

```bash
# Before Git is initialized:
python scripts/check_public_release.py --standalone

# After files have been added to Git:
python scripts/check_public_release.py --working-tree
PYTHONPYCACHEPREFIX=/tmp/public-release-pycache \
  python -m compileall -q code qa_generation pop_generation baselines scripts
bash -n run_all.sh
bash -n qa_generation/run_vllm.sh
bash -n pop_generation/run_pop_vllm.sh
```

## GitHub settings

- Add the Apache-2.0 license identifier in the repository description.
- Enable the artifact-request issue form.
- Disable large binary uploads unless they are part of a deliberate release.
- Create a tagged release for each paper version.
- If an external artifact archive is later published, record its DOI, checksum,
  archive size, and directory layout in `docs/DATA_AND_ARTIFACTS.md`.
