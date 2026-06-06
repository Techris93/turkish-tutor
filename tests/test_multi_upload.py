import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from auth_storage import configure_database, drop_db, init_db
from content_intelligence import ExtractionError


class MultiUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.database_url = f"sqlite:///{Path(self.tmpdir.name) / 'test_multi.sqlite3'}"
        os.environ["DATABASE_URL"] = self.database_url
        configure_database(self.database_url)
        drop_db()
        init_db()
        self.client = TestClient(api.app)

    def tearDown(self):
        os.environ.pop("DATABASE_URL", None)
        self.tmpdir.cleanup()

    @patch("api.ask_llm", new_callable=AsyncMock)
    def test_study_single_file_legacy_compatible(self, mock_ask):
        mock_ask.return_value = "Study Note for Single File"
        
        with patch("api.extract_vocabulary_items", return_value=[]):
            response = self.client.post(
                "/api/study",
                data={"level": "A1", "target_language": "English"},
                files={"file": ("legacy.txt", b"Merhaba, nasilsiniz?", "text/plain")},
            )
            
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_type"], "text-file")
        self.assertEqual(payload["source_label"], "legacy.txt")
        self.assertIn("Merhaba", payload["preview"])

    @patch("api.ask_llm", new_callable=AsyncMock)
    def test_study_multiple_files_aggregated(self, mock_ask):
        mock_ask.return_value = "Aggregated Study Note"
        
        with patch("api.extract_vocabulary_items", return_value=[]):
            response = self.client.post(
                "/api/study",
                data={"level": "A1", "target_language": "English"},
                files=[
                    ("files", ("first.txt", b"Bu ilk dosya.", "text/plain")),
                    ("files", ("second.txt", b"Bu ikinci dosya.", "text/plain")),
                ],
            )
            
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        # Different filenames mean source_label is joined by comma
        self.assertEqual(payload["source_label"], "first.txt, second.txt")
        self.assertEqual(payload["source_type"], "text-file")  # Both are text files, so specific type matches
        
        # Verify text boundary preservation
        self.assertIn("--- File: first.txt ---", payload["preview"])
        self.assertIn("Bu ilk dosya.", payload["preview"])
        self.assertIn("--- File: second.txt ---", payload["preview"])
        self.assertIn("Bu ikinci dosya.", payload["preview"])

    @patch("api.ask_llm", new_callable=AsyncMock)
    @patch("api.extract_text_from_file_details")
    def test_study_multiple_mixed_types(self, mock_extract, mock_ask):
        mock_ask.return_value = "Mixed Type Note"
        mock_extract.side_effect = lambda path: (
            ("Resim metni.", "image", "") if Path(path).suffix.lower() == ".png"
            else ("Yazili metin.", "text-file", "")
        )
        
        with patch("api.extract_vocabulary_items", return_value=[]):
            response = self.client.post(
                "/api/study",
                data={"level": "A2", "target_language": "English"},
                files=[
                    ("files", ("document.txt", b"Yazili metin.", "text/plain")),
                    ("files", ("photo.png", b"dummy png binary data", "image/png")),
                ],
            )
            
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source_label"], "document.txt, photo.png")
        self.assertEqual(payload["source_type"], "mixed")  # Mixed type should be returned

    def test_study_multi_rejects_unsupported(self):
        response = self.client.post(
            "/api/study",
            data={"level": "A1", "target_language": "English"},
            files=[
                ("files", ("good.txt", b"Guzel gun.", "text/plain")),
                ("files", ("malicious.exe", b"binary data", "application/octet-stream")),
            ],
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported input file type", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
