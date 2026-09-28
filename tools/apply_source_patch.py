"""Apply a recorded unified source patch strictly into the build directory.

Reads: upstream file, unified patch. Every context/removal line must match exactly.
"""
import re
import sys
from pathlib import Path

source, patch, target = map(Path, sys.argv[1:])
original = source.read_text().splitlines(keepends=True)
output = []
position = 0
for line in patch.read_text().splitlines(keepends=True)[2:]:
    if line.startswith("@@"):
        match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if match is None:
            raise RuntimeError("malformed patch header")
        start = int(match[1]) - 1
        if start < position:
            raise RuntimeError("overlapping patch hunks")
        output.extend(original[position:start])
        position = start
    elif line.startswith((" ", "-")):
        if position >= len(original) or original[position] != line[1:]:
            raise RuntimeError(f"patch does not match {source}:{position + 1}")
        if line[0] == " ":
            output.append(original[position])
        position += 1
    elif line.startswith("+"):
        output.append(line[1:])
    else:
        raise RuntimeError("unsupported patch line")
output.extend(original[position:])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("".join(output))
