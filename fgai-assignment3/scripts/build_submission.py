"""Build the required Word cover document; export it to PDF using Word Save As."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE

ROOT = Path(__file__).resolve().parents[1]


def hyperlink(paragraph, label, url):
    rel = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), rel)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "155799")
    properties.append(color)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    link.append(run)
    paragraph._p.append(link)


def build(allow_pending=False):
    details = json.loads((ROOT / "submission/details.json").read_text(encoding="utf-8"))
    url = details["github_url"].strip()
    if not url and not allow_pending:
        raise SystemExit("Add your real GitHub repository URL to submission/details.json first. Use --allow-pending only for a working copy.")
    if url:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "github.com" or len(parsed.path.strip("/").split("/")) != 2:
            raise SystemExit("github_url must be an HTTPS GitHub owner/repository URL")
    doc = Document()
    # The bundled default template can contain a blue border in its Title style.
    # Remove paragraph borders from both styles and content for a plain cover.
    for element in (doc.styles.element, doc.element):
        for border in list(element.iter(qn("w:pBdr"))):
            border.getparent().remove(border)
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(.85)
    section.left_margin = section.right_margin = Inches(.9)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.12
    for style_name in ("Title", "Subtitle", "Heading 1", "Heading 2"):
        doc.styles[style_name].font.color.rgb = RGBColor(0, 0, 0)
    doc.styles["Title"].font.size = Pt(24)
    doc.styles["Title"].font.bold = True
    doc.styles["Heading 1"].font.size = Pt(12)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(14)
    doc.add_paragraph("English to French Machine Translation", "Title")
    doc.add_paragraph("Foundations of Generative AI ICT4403", "Subtitle")
    doc.add_paragraph("Assignment 3", "Subtitle")
    doc.add_paragraph("Repository submission for a comparison of encoder-decoder and decoder-only translation systems on a shared English-French test set.")
    for heading, value in (("Student name", details["student_name"]),
                           ("Registration number", details["registration_number"]),
                           ("Assignment title", details["assignment_title"]),
                           ("Autoregressive text generation system", details["system_title"])):
        doc.add_paragraph(heading, "Heading 1")
        doc.add_paragraph(value)
    doc.add_paragraph("Dataset", "Heading 1")
    doc.add_paragraph(details["dataset_name"] + ". English to French; snapshot dated 7 April 2019.")
    hyperlink(doc.add_paragraph(), details["dataset_url"], details["dataset_url"])
    p = doc.add_paragraph("Download: ")
    hyperlink(p, "Public English-French dataset archive", details["dataset_download_url"])
    doc.add_paragraph("GitHub repository", "Heading 1")
    if url:
        hyperlink(doc.add_paragraph(), url, url)
    else:
        doc.add_paragraph("[Add the GitHub repository URL before final submission]")
    doc.core_properties.author = details["student_name"]
    doc.core_properties.title = "ICT4403 Assignment 3 Machine Translation"
    name = re.sub(r"[^A-Za-z0-9]+", "_", details["student_name"]).strip("_")
    destination = ROOT / "submission" / f'{name}_{details["registration_number"]}_FGAI_Assignment3.docx'
    doc.save(destination)
    print(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-pending", action="store_true")
    build(parser.parse_args().allow_pending)
