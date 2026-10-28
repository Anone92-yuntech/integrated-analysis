"""Public-clone input selection, without opening native dialogs in tests."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import integrated_app as app


class InputSelectionTests(unittest.TestCase):
    def test_no_sibling_project_is_required_by_parser(self):
        self.assertIsNone(app.parse_args([]).input)

    def test_headless_requires_explicit_input(self):
        with patch.object(app, 'choose_input_path') as chooser, contextlib.redirect_stderr(io.StringIO()) as error:
            self.assertEqual(app.main(['--headless']), 1)
        chooser.assert_not_called()
        self.assertIn('--input is required', error.getvalue())

    def test_cancelled_picker_does_not_load_models(self):
        with patch.object(app, 'choose_input_path', return_value=None), patch.object(app, 'YOLO') as model:
            self.assertEqual(app.main([]), 0)
        model.assert_not_called()

    def test_picker_returns_video_and_always_destroys_dialog(self):
        import tkinter
        import tkinter.filedialog
        fake = MagicMock()
        with patch.object(tkinter, 'Tk', return_value=fake), patch.object(
                tkinter.filedialog, 'askopenfilename', return_value='example.mp4'):
            self.assertEqual(app.choose_input_path(), Path('example.mp4').resolve())
        fake.withdraw.assert_called_once()
        fake.destroy.assert_called_once()

    def test_cancel_returns_none(self):
        import tkinter
        import tkinter.filedialog
        fake = MagicMock()
        with patch.object(tkinter, 'Tk', return_value=fake), patch.object(
                tkinter.filedialog, 'askopenfilename', return_value=''):
            self.assertIsNone(app.choose_input_path())
        fake.destroy.assert_called_once()
