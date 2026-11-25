"""Dependency-free checks for the documentation link gate."""

import tempfile
import unittest
from pathlib import Path

from scripts.check_local_links import broken_links, destinations


class LocalLinksTest(unittest.TestCase):
    def test_inline_images_references_and_fences(self):
        source = """[page](../guide.md#section) ![plot](assets/chart.svg)
[reference]: <other file.md>
```md
[example](missing.md)
```
"""
        self.assertEqual(
            list(destinations(source)),
            ["../guide.md#section", "assets/chart.svg", "other file.md"],
        )

    def test_resolves_relative_and_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "guide.md").write_text(
                "[home](../README.md) [missing](absent.md) [web](https://example.com)",
                encoding="utf-8",
            )
            (root / "README.md").write_text("[guide](docs/guide.md#intro)", encoding="utf-8")
            self.assertEqual(list(broken_links(root)), ["docs/guide.md: absent.md"])


if __name__ == "__main__":
    unittest.main()
