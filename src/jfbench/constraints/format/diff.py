import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class DiffFormatError(ValueError):
    """Raised when the text is not a valid unified git diff."""


# Matches hunk headers of the form:
# @@ -start,len +start,len @@ optional heading
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffFormatConstraint(ConstraintGroupMixin):
    """
    Constraint that checks whether the given text is a (reasonably) strict
    unified git diff in the default `git diff` format.

    - Supports multiple files (expects `diff --git` headers).
    - Validates presence and ordering of `---` / `+++` file headers.
    - Parses hunks (`@@ ... @@`) and validates their old/new line counts.
    """

    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned = self._strip_code_fence(value)
        text = cleaned.rstrip("\n")
        try:
            has_hunk = self._parse_unified_diff(text)
        except DiffFormatError as e:
            reason = f"[Diff Format] {e}"
            logger.info(reason)
            return False, reason

        if not has_hunk:
            reason = "[Diff Format] No hunks (@@) found in diff."
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "Git の unified diff 形式で出力してください。",
            "git diff のデフォルト出力と同じ unified diff 形式で結果を記述してください。",
            "変更点を git の unified diff スタイルで示してください。",
            "パッチ形式の unified diff を生成してください。",
            "git diff の unified diff 出力フォーマットを用いてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "Gitのdiff形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "Git の unified diff 形式になるように整形してください。"

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        stripped = value.strip()
        match = re.match(r"\A```[^\n]*\n(.*)\n```\s*\Z", stripped, re.DOTALL)
        if match is None:
            return value
        return match.group(1)

    # ==========
    # Parser core
    # ==========

    def _parse_unified_diff(self, text: str) -> bool:
        """
        Parse the whole unified git diff.

        Raises:
            DiffFormatError: if the text is not a valid unified git diff.

        Returns:
            bool: True if at least one hunk exists, False otherwise.
        """
        if not text.strip():
            raise DiffFormatError("Empty output.")

        lines = text.splitlines()
        i = 0
        n = len(lines)

        has_diff_block = False
        has_any_hunk = False

        # Allow diffs that start directly with file headers (no diff --git line)
        while i < n and not lines[i].strip():
            i += 1
        if i < n and lines[i].startswith("--- "):
            _, has_hunk = self._parse_file_diff(lines, i)
            return has_hunk

        while i < n:
            line = lines[i]

            # Skip empty lines between diff blocks
            if not line.strip():
                i += 1
                continue

            if line.startswith("diff --git "):
                has_diff_block = True
                i, file_has_hunk = self._parse_file_diff(lines, i + 1)
                has_any_hunk = has_any_hunk or file_has_hunk
                continue
            if line.startswith("--- "):
                has_diff_block = True
                i, file_has_hunk = self._parse_file_diff(lines, i)
                has_any_hunk = has_any_hunk or file_has_hunk
                continue

            # Any non-empty line outside a diff block is considered invalid
            raise DiffFormatError(f"Unexpected line outside of any 'diff --git' block: {line!r}")

        if not has_diff_block:
            raise DiffFormatError("Missing unified diff headers.")

        return has_any_hunk

    def _parse_file_diff(self, lines: list[str], i: int) -> tuple[int, bool]:
        """
        Parse a single 'diff --git ...' file block.

        Args:
            lines (List[str]): All lines of the diff.
            i (int): Index of the line immediately after 'diff --git ...'.

        Returns:
            tuple[int, bool]: (next_index, file_has_hunk)
                next_index: index of the next line to read (may be a new
                            'diff --git' or EOF).
                file_has_hunk: True if at least one hunk was parsed for this file.
        """
        n = len(lines)
        saw_old_header = False
        saw_new_header = False
        file_has_hunk = False

        # Parse file header lines until we reach hunks or next diff block
        while i < n:
            line = lines[i]

            # Start of the next diff block => end of this file block
            if line.startswith("diff --git "):
                break

            # Known file-header-related lines
            if line.startswith(
                (
                    "index ",
                    "new file mode ",
                    "deleted file mode ",
                    "old mode ",
                    "new mode ",
                    "similarity index ",
                    "rename from ",
                    "rename to ",
                    "--- ",  # old file header
                    "+++ ",  # new file header
                )
            ):
                if line.startswith("--- "):
                    saw_old_header = True
                    i += 1
                    if i >= n or not lines[i].startswith("+++ "):
                        raise DiffFormatError("Expected '+++ ' header after '--- ' header.")
                    saw_new_header = True
                    i += 1

                    # After '---' and '+++', we expect one or more hunks
                    i, has_hunk = self._parse_hunks(lines, i)
                    file_has_hunk = file_has_hunk or has_hunk
                    # For a typical git diff, hunks for this file are contiguous,
                    # so we stop here and let the outer loop handle the next block.
                    break
                elif line.startswith("+++ "):
                    # Seeing '+++' without a preceding '---' is suspicious
                    raise DiffFormatError("Saw '+++ ' header without preceding '--- ' header.")
                else:
                    # Valid header line, just move on
                    i += 1
                    continue

            elif not line.strip():
                # Allow blank lines in the header section
                i += 1
            else:
                # Unknown / unexpected header line
                raise DiffFormatError(f"Unexpected line in file header: {line!r}")

        if not saw_old_header or not saw_new_header:
            # Note: git can output diff blocks without hunks (e.g., rename only),
            # but for our "patch-like diff" constraint we require both headers.
            raise DiffFormatError("Missing '---'/'+++' file headers.")

        return i, file_has_hunk

    def _parse_hunks(self, lines: list[str], i: int) -> tuple[int, bool]:
        """
        Parse consecutive hunks after the file headers.

        Args:
            lines (List[str]): All lines of the diff.
            i (int): Index of the first line after '---'/'+++' headers.

        Returns:
            tuple[int, bool]: (next_index, has_hunk)
                next_index: index of the next line to read (may be 'diff --git',
                            a new file header, or EOF).
                has_hunk: True if at least one hunk was parsed.
        """
        n = len(lines)
        has_hunk = False

        while i < n:
            line = lines[i]

            # A new diff block means the end of hunks for this file
            if line.startswith("diff --git "):
                break

            # A new hunk starts here
            if line.startswith("@@ "):
                has_hunk = True
                match = _HUNK_HEADER_RE.match(line)
                if not match:
                    raise DiffFormatError(f"Malformed hunk header: {line!r}")

                old_start = int(match.group(1))
                old_len = int(match.group(2) or "1")
                new_start = int(match.group(3))
                new_len = int(match.group(4) or "1")

                # Sanity checks on the positions (can be relaxed if needed)
                if old_start < 0 or new_start < 0:
                    raise DiffFormatError("Negative start in hunk header.")

                # Count the actual lines in the hunk body
                i += 1
                old_count = 0
                new_count = 0

                while i < n:
                    line_content = lines[i]

                    # A new hunk, new diff block, or file header marks the end of this hunk
                    if (
                        line_content.startswith("@@ ")
                        or line_content.startswith("diff --git ")
                        or line_content.startswith("--- ")
                    ):
                        break

                    if not line_content:
                        # In unified diff, lines in the hunk should always have
                        # a prefix: ' ', '+', '-', or '\' for the special line.
                        raise DiffFormatError("Empty line inside hunk without prefix.")

                    prefix = line_content[0]

                    if prefix == " ":
                        old_count += 1
                        new_count += 1
                    elif prefix == "-":
                        old_count += 1
                    elif prefix == "+":
                        new_count += 1
                    elif line_content.startswith("\\ No newline at end of file"):
                        # This is a special line and does not count as content
                        pass
                    else:
                        raise DiffFormatError(f"Unexpected line inside hunk: {line_content!r}")

                    i += 1

                # Check that the counts match the hunk header
                if old_count > old_len or new_count > new_len:
                    raise DiffFormatError(
                        "Hunk line count exceeds header: "
                        f"header -{old_len},+{new_len} "
                        f"but saw -{old_count},+{new_count}."
                    )

                # i now points at the next hunk, diff block, file header, or EOF
                continue

            # Allow blank lines between hunks
            if not line.strip():
                i += 1
                continue

            # Any other content here is unexpected
            raise DiffFormatError(
                f"Unexpected line inside file diff (expecting '@@' or blank): {line!r}"
            )

        return i, has_hunk
