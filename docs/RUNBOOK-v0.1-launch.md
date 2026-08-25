# v0.1 Launch Runbook

This document turns launch day into an execution problem instead of a
judgment problem. Every judgment it encodes — ordering, failure modes,
verification, rollback honesty — was made and frozen ahead of time. If
you are reading this on flip day and find yourself *deciding* something,
stop: you are in the wrong document, and the decision belongs to the
project maintainer, recorded before proceeding.

Two standing rules govern everything below:

- **Visibility flips, weight-host changes, the PyPI release, and every
  public announcement are human acts.** Agents prepare, verify, and
  record; a human performs the irreversible click. This is stated again
  on each step it applies to.
- **No agent ever posts publicly** — not an announcement, not an issue
  reply, not a comment. Agents may draft; a human posts.

Any halt, at any point, is written into the maintainers' state log
before anyone walks away. An undocumented half-flipped state is the one
outcome this runbook exists to prevent.

The verification companion is [`scripts/flip-verify.py`](../scripts/flip-verify.py)
— one subcommand per verifiable step, described in §7.

---

## 1. Preconditions

Every box below must be checked before step 1 of the flip sequence is
performed. These are binary: a box that is "mostly done" is unchecked.

- [ ] **Sign-off verdicts written, per component.** A written
  approval exists for every component version the launch registry
  will serve (identity, each texture adapter, interpreter default,
  body-rig, assembly-assets, hair provider), each naming the exact
  artifact hash it approves.
  *Check:* the verdict document exists in the maintainers' workspace and
  every `name@version` the launch snapshot will declare appears in it
  with a hash.
- [ ] **The served stack matches the verdicts.** The review app's live
  component index resolves exactly the approved versions — what was
  inspected is what ships.
  *Check:* `GET /v0/components` on the review deployment; compare
  `name`/`version` rows against the verdict list. Zero deviations.
- [ ] **Weights staged to the private weight host and hash-verified.**
  Every artifact the launch registry will reference is uploaded to its
  private hosting repository at a pinned revision, and the uploaded
  bytes hash to the pinned values.
  *Check:* `scripts/flip-verify.py step1 --authenticated` against a
  snapshot draft carrying the private sources — every artifact fetches
  and sha256-verifies. (Pre-flip this uses the configured
  `CHARACTER_FACTORY_AUTH_TOKEN`; see §7.)
- [ ] **Cold-cache resolution end-to-end on a second machine.** On a
  machine that is not the build machine, with an empty
  `CHARACTER_FACTORY_HOME`, the private-form registry resolves,
  downloads, verifies, and an example character assembles to a valid
  GLB.
  *Check:* `scripts/flip-verify.py step2 --authenticated` on that
  machine exits 0.
- [ ] **The macOS assemble-only test passed.** On the Apple-silicon
  laptop (CPU only, no CUDA): fresh venv, `pip install -e .` from a
  checkout, then the cold-cache assemble above. This is the proof of
  the README's "assembly runs anywhere" claim — wheels for every base
  dependency, no GPU touched.
  *Check:* `scripts/flip-verify.py step2 --authenticated` exits 0 on
  that laptop.
- [ ] **Repo green at the flip sha.** The full test suite passes at the
  exact commit that will become public — including
  `tests/registry/test_examples_resolve.py`, which pins every committed
  example to the packaged snapshot.
  *Check:* `python -m pytest tests` at the flip sha: 0 failures. Record
  the sha; it is the `--expect-sha` input to step 3's verification.
- [ ] **Full-history provenance and secrets audit re-run.** The
  2026-08-24 full-history audit (recorded in the maintainers' provenance
  log, `PROVENANCE.log`, entry "FULL-HISTORY AUDIT (pre-push, first
  remote)") is the baseline: every blob in every revision, all commit
  messages, all author identities, key-shaped-string sweep, and a
  fixed-string search for every live credential. Re-run the same sweep
  over everything committed since that baseline, and log the result as
  a new entry.
  *Check:* the new `PROVENANCE.log` entry exists, dated within 48 hours
  of the flip, reading zero real hits.
- [ ] **Model cards written.** *To produce — owner: maintainer.* One
  card per published weight repository (contents are the maintainer's
  voice; agents do not author public model claims).
  *Check:* the card files exist in each staging repository before it
  flips public.
- [ ] **PyPI name confirmed and the release path tested.** As of
  2026-08-25, `character-factory` has no distribution on PyPI
  (`pip install character-factory` → "no matching distribution"); PyPI
  does not reserve names without an upload, so the claim happens at
  release. *To produce — owner: maintainer:* a full rehearsal against
  test.pypi.org — build sdist and wheel from the flip sha, upload with
  the maintainer's account, install from the test index into a fresh
  venv, run `character-factory validate` on an example.
  *Check:* the rehearsal completed within a week of the flip; re-verify
  the name is still unclaimed on flip day *before* step 1 (a squatted
  name changes the plan and is a maintainer decision, not a runbook
  branch).
- [ ] **Announcement drafts exist as files.** *To produce — owner:
  maintainer.* Whatever channels will carry the announcement, the text
  exists on disk before the sequence opens, so step 6 is a paste, not a
  composition.
  *Check:* the draft files exist in the maintainers' workspace.

---

## 2. The flip sequence

Ordering rationale: publish the *least discoverable, most reversible*
things first and let each step's verification depend only on steps
already done. Weights go public before anything points at them (a public
weight repo nobody links to is quiet, and re-privating it is cheap);
the registry's public form can then be verified end-to-end while the
code repo is still private; the repo flip — the effectively one-way
step — happens only after everything it references already works; the
package release happens only after the repo it links to is public; the
tag and announcements come last because they are pure pointers at a
state already verified. At every boundary the world is in a state that
is safe to stop in.

### Step 1 — weight repositories public

- **Action:** flip each staging weight repository from private to
  public on the weight host, in any order. Model cards (precondition)
  are already in place.
- **Who:** the **maintainer, by hand**, in the host's UI or CLI under
  their own account. Agents verify only.
- **Verify:** `scripts/flip-verify.py step1` (anonymous). Expected: one
  `ok` line per artifact, byte count and `sha256 verified`, ending
  `all N declared artifacts fetch and verify (anonymous)`.
- **Stop-rule:** any `FAIL` line halts the sequence — a 401/403 means a
  repo is still private; a hash or byte-count mismatch means the staged
  upload is not the approved artifact, and **nothing else happens until
  the mismatch is explained in writing**. A mismatch here is the
  cheapest it will ever be; downstream it becomes a user-facing
  integrity failure.

### Step 2 — registry index to its public form

- **Action:** commit the update to
  `src/character_factory/registry/data/registry-snapshot.json` filling
  in each entry's `source` (public repository + pinned revision) and
  its `artifacts` list (path, sha256, bytes) — the values verified in
  step 1. Push to the (still-private) org repo. This commit is the
  **flip sha**.
- **Who:** an agent may author and push the commit (it is code, gated
  like all code); the **maintainer reviews the diff** — this file is
  the trust root users' hash verification hangs from.
- **Verify:** `scripts/flip-verify.py step2` (anonymous) on the flip
  sha: cold cache → packaged snapshot → `body-rig` and
  `assembly-assets` download and verify → `marathon-runner` example
  assembles to a >1 MB GLB. Then `python -m pytest tests` once more at
  the flip sha.
- **Stop-rule:** resolution or assemble failure halts. The failure is
  in data committed one step ago against artifacts published one step
  ago — fix at the source (a corrected snapshot commit), never by
  hand-editing anything on the weight host.

### Step 3 — org repository public

- **Action:** flip `github.com/character-factory/character-factory`
  from private to public, at the flip sha.
- **Who:** the **maintainer, by hand**, in the repository settings.
  This is the effectively one-way step (§3). Agents verify only.
- **Verify:** `scripts/flip-verify.py step3 --expect-sha <flip-sha>`
  from a context holding no GitHub credentials. Expected: anonymous
  clone succeeds, `LICENSE` and `NOTICE` present, public HEAD matches
  the flip sha.
- **Stop-rule:** clone refusal means the flip did not take — retry the
  setting, nothing else changed. A *wrong sha* means unreviewed commits
  are public: halt, and the maintainer decides between flipping back
  immediately (minutes matter; see §3) and accepting the extra commits
  after reading them.

### Step 4 — PyPI release

- **Action:** from the flip sha: build sdist + wheel
  (`python -m build`), upload to PyPI (`twine upload dist/*`) as
  version `0.1.0`. Then commit the README release edit (remove the
  pre-release banner) and push — the repo is public now; this commit
  goes through the gate like every other.
- **Who:** the **upload is the maintainer's act** under their own PyPI
  account (this is also what claims the name). Agents may build the
  artifacts and verify; agents never hold or use the PyPI credential.
- **Verify:** `scripts/flip-verify.py step4`. Expected: fresh venv,
  `pip install character-factory` succeeds, `character-factory
  validate` passes on an example, `preflight` runs and names its
  causes, and the installed package assembles the example from a cold
  cache.
- **Stop-rule:** an install or import failure halts *this step only*
  (the repo being public without a package is a safe state, §4). Do not
  fix a broken 0.1.0 by re-upload — PyPI files are immutable; the fix
  is `0.1.1` (§3).

### Step 5 — tag and release

- **Action:** tag `v0.1.0` on the README release commit, push the tag,
  create the host release entry pointing at it.
- **Who:** an agent may prepare the tag and release text; the
  **maintainer pushes the tag and publishes the release**.
- **Verify:** `git ls-remote --tags origin` shows `v0.1.0` at the
  expected sha; the release page renders.
- **Stop-rule:** none that halts the world — a wrong tag is deleted and
  re-pushed before announcements, and that is the reason the tag
  precedes them.

### Step 6 — announcements

- **Action:** the maintainer posts the pre-written drafts
  (precondition) to their channels.
- **Who:** the **maintainer only**. No agent ever posts publicly — this
  is the standing rule, not a step property.
- **Verify:** links in the posted text resolve: the repo, the release,
  `pip install character-factory`.
- **Stop-rule:** announcements are the one step that cannot be unsaid.
  If anything upstream feels wrong, stop *before* this step; everything
  else about the launch will wait overnight without cost (§4).

---

## 3. Rollback honesty

| Step | Undoable? | What undoing truly costs | Structural mitigation |
|---|---|---|---|
| 1. Weights public | Yes — re-flip private | Anyone who fetched in the window keeps the bytes; mirrors are possible but unlikely in an unannounced window measured in minutes-to-hours | Pinned hashes: no matter who mirrors what, a client verifies sha256 against the registry — substituted or corrupted copies fail loudly |
| 2. Public-form snapshot commit | Yes — revert commit | Nothing; the repo is still private at this point | Ordinary git history |
| 3. Repo public | **Treat as one-way** | Forks, clones, and archive crawlers can capture the tree within minutes; re-privating does not un-publish and (with forks) may not even be possible | The born-public discipline: every commit in this repository's history was written and gated as public from inception — there is nothing in the history whose exposure is a new event |
| 4. PyPI 0.1.0 | No — files are immutable | A broken 0.1.0 is yanked (`pip` stops selecting it, but it remains downloadable by pin) and superseded by 0.1.1; the version number is spent | The release-path rehearsal against test.pypi.org (precondition) exists to make this rollback never needed |
| 5. Tag / release | Mostly — tags can be deleted and re-pushed | Clones made in the window keep the old tag; harmless before announcements | Tag precedes announcements by design |
| 6. Announcements | **No** | Cannot be unsaid | It is last for exactly this reason |

The same honesty applies to weights after launch — see the bold rule in
§5.

---

## 4. Abort states

Rule zero: **every halt is written into the maintainers' state log
before anyone walks away** — which step completed, what the verifier
printed, what the next action is. All of the following states are safe
to leave overnight:

- **Halt after step 1:** public weight repos, nothing pointing at them.
  Quiet and stable. Overnight: fine.
- **Halt after step 2:** as above, plus a private repo whose snapshot
  references public weights. Fully consistent. Overnight: fine.
- **Halt after step 3:** the repo is public, README carries the
  pre-release banner ("under construction, not yet on PyPI") — which is
  exactly true in this state. An early visitor can clone, read, build
  from source, and resolve weights. Overnight: fine; this is a soft
  launch, not a broken one.
- **Halt after step 4:** package installable, README banner possibly
  not yet removed (harmless: it promises less than reality delivers).
  Overnight: fine.
- **Halt after step 5:** tagged, released, unannounced. The classic
  quiet launch. Overnight: fine — some projects stop here on purpose.

There is no state in this sequence that requires a rushed 2 a.m.
continuation. When in doubt: log, stop, sleep.

---

## 5. The first 48 hours

**Intake:** GitHub issues on the org repository are the only intake
channel. Agents may triage, reproduce, label, and *draft* replies; a
human posts every public word (standing rule).

**Two classes, triaged on sight:**

- **Fix-now** — anything that breaks the first five minutes for
  everyone: `pip install` failure on a first-class platform, an example
  that does not validate or resolve, a registry artifact that 404s or
  fails hash verification, `character-factory make`/`assemble` crashing
  on the documented path. These pre-empt everything; the fix ships as
  0.1.1 (code) or a registry/snapshot correction (data), verified with
  the same `flip-verify.py` steps before the fix is announced fixed.
- **Log** — everything else: output-quality reports, feature requests,
  platform requests, puzzling-but-workaroundable behavior. Labeled,
  acknowledged (by a human), batched. Nothing in this class justifies
  touching published weights or rushing a release in the first 48
  hours.

**The weights rule, which has no exceptions:**

> **A defective published weight is fixed by publishing a NEW component
> version, never by re-uploading over an existing one. Pinned hashes
> verify forever — a re-upload does not "fix" anyone; it breaks hash
> verification for everyone.**

The registry is designed around this: a new version is a new entry with
new pins, old character files keep resolving the version they recorded,
and the interpreter/defaults move forward by resolution, not mutation.

---

## 6. Explicitly out of scope

- **The hosted (cloud) launch.** Own clock, own runbook. Nothing here
  gates on it; nothing in it gates on this document beyond the repo
  being public.
- **The engine-integration (Unity) package publication.** Its own
  small checklist with its own owner. It must not gate this
  sequence, and this sequence does not verify it.
- **All judgment calls.** Pricing, positioning, what to say in
  announcements, whether a squatted PyPI name changes the package name,
  whether a halt becomes an abort — maintainer decisions, made outside
  this document and recorded in the state log.

---

## 7. The verify script

[`scripts/flip-verify.py`](../scripts/flip-verify.py) — run from a
checkout with the package importable. One subcommand per verifiable
step; exit 0 is the only pass.

| Subcommand | Verifies | Flip step |
|---|---|---|
| `step1 [--authenticated]` | every registry-declared artifact fetches (anonymously by default) and matches its pinned sha256 and byte count | 1 |
| `step2 [--authenticated]` | cold-cache resolution (`CHARACTER_FACTORY_HOME` pointed at an empty directory) + the marathon-runner example assembles to a valid GLB on CPU | 2 |
| `step3 [--expect-sha SHA]` | anonymous `git clone` succeeds, `LICENSE`/`NOTICE` present, public HEAD is the flip sha | 3 |
| `step4` | fresh venv → `pip install character-factory` → `validate`, `preflight`, and a cold-cache assemble through the *installed* package | 4 |

`--authenticated` applies the configured
`CHARACTER_FACTORY_REGISTRY_URL`/`CHARACTER_FACTORY_AUTH_TOKEN`
credentials instead of anonymous access, turning steps 1–2 into the
pre-flip staging verification used by the preconditions. The
anonymous forms are the launch-day truth.

**Pre-flip behavior, tested 2026-08-25** — before the flip the anonymous
forms must fail, and each was run against the current private state to
assert it fails for exactly the right reason:

- `step1` → `FAIL … not published yet: its registry entry has no source
  repository` for hash-pinned entries, `no artifacts declared` for the
  rest. (Any *other* failure — e.g. a hash mismatch on a fetchable
  artifact — would be a real finding, pre-flip or not.)
- `step2` → `FAIL cold-cache resolution: component body-rig@1.1.0 is
  not published yet…`
- `step3` → `FAIL anonymous clone refused: fatal: could not read
  Username…`
- `step4` → `FAIL pip install failed: ERROR: No matching distribution
  found for character-factory`

One trap found while testing and now guarded in the script: a
`PYTHONPATH` pointing into a checkout makes `pip` report the package
"already satisfied" without installing anything — `step4` scrubs
`PYTHONPATH` from every subprocess so it can only pass against what
PyPI actually serves.
