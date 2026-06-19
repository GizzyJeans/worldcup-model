"""Install the worldcup-betting skill into this machine's Claude config.

Run once per machine after cloning:  python install_skill.py

Copies skills/worldcup-betting/SKILL.md into ~/.claude/skills/, substituting
{{ROOT}} with this clone's absolute path so the commands point at the right
place. New Claude Code sessions on this machine will then list the skill.
"""

from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> None:
    template = (ROOT / "skills" / "worldcup-betting" / "SKILL.md").read_text(encoding="utf-8")
    content = template.replace("{{ROOT}}", ROOT.as_posix())

    dest_dir = pathlib.Path.home() / ".claude" / "skills" / "worldcup-betting"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "SKILL.md"
    dest.write_text(content, encoding="utf-8")
    print(f"installed skill -> {dest}")
    print("Start a new Claude Code session for it to appear in the skill list.")


if __name__ == "__main__":
    main()
