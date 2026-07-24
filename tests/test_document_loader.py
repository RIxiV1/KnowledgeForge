"""Tests for document_loader.load_file across file types (offline, no Ollama)."""
import fitz  # PyMuPDF
import pytest

import document_loader


def _make_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    known = "KnowledgeForge unit test marker sentence."
    _make_pdf(pdf_path, known)

    docs = document_loader.load_file(str(pdf_path))

    assert len(docs) >= 1
    first = docs[0]
    assert "KnowledgeForge" in first.page_content
    assert first.metadata["source"] == "sample.pdf"
    assert first.metadata["file_type"] == "pdf"
    assert first.metadata["page"] == 1


def test_load_txt(tmp_path):
    txt_path = tmp_path / "note.txt"
    txt_path.write_text("hello world from a text file", encoding="utf-8")

    docs = document_loader.load_file(str(txt_path))

    assert len(docs) == 1
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["source"] == "note.txt"
    assert "hello world" in docs[0].page_content


def test_load_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,amount,year\nAlice,10,2023\nBob,20,2024\n", encoding="utf-8")

    docs = document_loader.load_file(str(csv_path))

    assert len(docs) >= 1
    assert docs[0].metadata["file_type"] == "csv"
    assert docs[0].metadata["source"] == "data.csv"
    assert "Alice" in docs[0].page_content


def test_load_docx_splits_by_heading(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Introduction", level=1)
    d.add_paragraph("KnowledgeForge is a local RAG app.")
    d.add_heading("Architecture", level=1)
    d.add_paragraph("It fuses vector search and BM25.")
    path = tmp_path / "design.docx"
    d.save(str(path))

    docs = document_loader.load_file(str(path))

    assert len(docs) == 2
    sections = {doc.metadata.get("section") for doc in docs}
    assert sections == {"Introduction", "Architecture"}
    assert all(doc.metadata["file_type"] == "docx" for doc in docs)


def test_load_docx_without_headings_is_single_blob(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Just a plain paragraph.")
    d.add_paragraph("Another one, still no headings.")
    path = tmp_path / "plain.docx"
    d.save(str(path))

    docs = document_loader.load_file(str(path))

    assert len(docs) == 1
    assert docs[0].metadata.get("section") is None


def test_load_pptx_one_doc_per_slide(tmp_path):
    pptx = pytest.importorskip("pptx")
    pres = pptx.Presentation()
    for title in ("Overview", "Roadmap"):
        slide = pres.slides.add_slide(pres.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = f"Content for {title}."
    path = tmp_path / "deck.pptx"
    pres.save(str(path))

    docs = document_loader.load_file(str(path))

    assert len(docs) == 2
    assert [doc.metadata.get("page") for doc in docs] == [1, 2]
    assert all(doc.metadata["file_type"] == "pptx" for doc in docs)


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "thing.xyz"
    bad.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError):
        document_loader.load_file(str(bad))
