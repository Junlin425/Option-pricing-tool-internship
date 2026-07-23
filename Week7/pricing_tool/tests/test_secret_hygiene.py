import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SENSITIVE_FILES = (
    ROOT / "Week1" / "scripts" / "test_alpha.py",
    ROOT / "Week1" / "scripts" / "test_fred.py",
    ROOT / "Week2" / "scripts" / "news_sentiment.py",
    ROOT / "Week1" / "scripts" / "api_configuration.ipynb",
)

LITERAL_API_KEY_PATTERN = re.compile(
    r"(?i)(?:api[_ -]?key|apikey|\bkey)\s*(?:=|:)\s*"
    r"(?:\{\s*)?[\"'][A-Za-z0-9]{8,}[\"']"
)


class SecretHygieneTests(unittest.TestCase):
    def test_sensitive_files_have_no_literal_api_key_assignments(self):
        for path in SENSITIVE_FILES:
            with self.subTest(path=path):
                content = path.read_text(encoding="utf-8").replace('\\"', '"')
                if LITERAL_API_KEY_PATTERN.search(content):
                    self.fail(f"Hard-coded API credential found in {path.name}")


if __name__ == "__main__":
    unittest.main()
