# Caveman — terse-style prompt/output compression

[Caveman](https://github.com/juliusbrussee/caveman) is a third-party
skill that strips filler, politeness, and redundant grammar from model
output to cut tokens by ~65% on verbose turns. Upstream's install path
targets Claude Code's plugin marketplace; below is the manual install
for claw.

## Install

Caveman's upstream uses `claude plugin install caveman@caveman`, which
claw doesn't support. Install the skill files manually instead:

```sh
# Clone upstream into a scratch dir (any location works; we copy out).
git clone --depth=1 https://github.com/juliusbrussee/caveman /tmp/caveman

# claw auto-discovers skills in .claw/skills/<name>/SKILL.md — user-level.
mkdir -p ~/.claw/skills/caveman
cp /tmp/caveman/skills/caveman/SKILL.md ~/.claw/skills/caveman/SKILL.md

# (repeat for any lite/full/ultra variants you want)
for v in caveman-lite caveman-full caveman-ultra; do
    [ -f /tmp/caveman/skills/$v/SKILL.md ] || continue
    mkdir -p ~/.claw/skills/$v
    cp /tmp/caveman/skills/$v/SKILL.md ~/.claw/skills/$v/SKILL.md
done

rm -rf /tmp/caveman
```

Project-scoped install is the same with `./.claw/skills/` instead of
`~/.claw/skills/`.

## Activate

Caveman upstream relies on a `SessionStart` hook to auto-enable itself
every turn — **claw doesn't implement `SessionStart` hooks**, so that
auto-activation does not port over. You invoke Caveman manually:

```
/caveman            # compress this turn
/caveman lite       # lighter compression
/caveman full       # aggressive
/caveman ultra      # maximum, near-telegram
```

The skill file does the work; no other claw config needed.

## How it pairs with the other efficiency layers

- **Caveman** compresses the *model's reply* (and optionally the user
  prompt). Saves tokens on the response side.
- **RTK** (see `../rtk/`) compresses *tool-call output* — the bytes flowing
  from `git status` / `cargo check` etc. back into the context.
- **Router scoreboard** + **prompt cache** (already in claw via the
  `router` / `memory` config blocks) cut cost by picking cheaper models
  for routine turns and re-using identical-prompt responses.

Stacked, the per-turn token cost can drop by a meaningful multiple; each
layer is independent and opt-in.

## Verify it's loaded

```
/skills
```

Should list `caveman` (and any variants you installed).
