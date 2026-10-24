# Log — reproducibility fix + second snapshot push (2026-10-25)

Trigger: user requirement (2026-10-25 00:09 CST) — the repo must
support FULL reproduction, and must contain the conversation record
and detailed code explanations.

Audit findings (checked against the pushed snapshot, not memory):
- README "What is in here" claimed the model-parameter .npz files,
  stage2_models.pt, and PNG figures were included, but the first
  snapshot push (commit 5b3b0c14) contains text files only: the
  upload connection rejects binaries, so 24 binary files were held
  back. The README statement was wrong for the GitHub copy and is
  corrected in this commit.
- conversation/conversation.md stopped at "push pending" and did not
  record the browser repo creation, the App authorisation, the first
  push, or the task1–task5 repos.
- No per-script detailed explanation document existed (scripts have
  docstrings; README has pipeline order; SKILL has frozen decisions).
- requirements.txt recorded tested versions but pinned only pin;
  numpy/scipy/matplotlib/torch were unpinned.
- Data download was manual (DATA.md links + MD5s, no script).

Fixes in this commit:
- README.md: "What is in here" split into GitHub-snapshot contents
  (text only) vs local-package-only binaries, with the regeneration
  path; pipeline step 1 now points at download_data.py.
- CODE.md (new): per-script purpose / inputs / outputs / key
  decisions for all 13 scripts (12 pipeline + download_data.py),
  plus the regeneration-chain argument.
- conversation/conversation.md: section 16 appended (repo creation,
  App authorisation, first push commit 5b3b0c14, task1–task5,
  reproducibility requirement and this fix); "Open decisions"
  updated (GitHub target resolved; license + stage-2 decision open).
- requirements.txt: recommended venv versions pinned
  (numpy==2.5.4, scipy==1.18.1, matplotlib==3.11.2, pin==4.1.0,
  torch==2.14.1+cpu via the CPU index); system-python alternative
  kept as a note.
- download_data.py (new): stdlib-only download + size/MD5 verify
  for the two data packs; extraction left manual (RAR v4); Videos.rar
  deliberately excluded.
- This log file.

Verification before push: new/changed files compile (py_compile),
manifest built from git ls-files with the same binary exclusions
(*.npz, *.pt, *.png) and sync-record exclusions, all entries UTF-8,
total under the 16 MiB snapshot limit. Push verification (commit
SHA, README + CODE.md read-back from main) is recorded in the sync
record next to this package and in MEMORY.md, not here, because this
log ships inside the pushed commit itself.

Binaries still NOT on GitHub after this push (same connection
limitation): 6 parameter .npz, stage2_models.pt, 17 PNGs. They are
all pipeline outputs and regenerate via download_data.py + the
README pipeline order; the local package retains them.
