"""Stub for python-pptx sufficient for tests that don't patch Presentation."""

import os


class _Font:
    def __init__(self):
        self.size = None
        self.name = None


class _Run:
    def __init__(self):
        self.font = _Font()


class _Paragraph:
    def __init__(self):
        self.text = ""
        self.level = 0
        self.alignment = 0
        self.runs = [_Run()]


class _TextFrame:
    def __init__(self):
        self._paragraphs = [_Paragraph()]

    def clear(self):
        self._paragraphs = [_Paragraph()]

    def add_paragraph(self):
        p = _Paragraph()
        self._paragraphs.append(p)
        return p

    @property
    def paragraphs(self):
        return self._paragraphs


class _Title:
    def __init__(self):
        self.text = ""


class _Placeholder:
    def __init__(self):
        self.text = ""
        self.text_frame = _TextFrame()


class _Notes:
    def __init__(self):
        self.notes_text_frame = type("_NTF", (), {"text": ""})()


class _Slide:
    def __init__(self):
        self.shapes = type("_Shapes", (), {"title": _Title()})()
        self.placeholders = {1: _Placeholder()}
        self.notes_slide = _Notes()


class _Slides:
    def __init__(self):
        self._slides = []

    def add_slide(self, _layout):
        s = _Slide()
        self._slides.append(s)
        return s


class Presentation:
    def __init__(self, *args, **kwargs):
        self.slide_layouts = [object(), object()]
        self.slides = _Slides()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"")

