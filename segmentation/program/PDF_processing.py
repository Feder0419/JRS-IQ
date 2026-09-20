import fitz
from pathlib import Path


def render_pdf(pdf_path: str, output_dir: str):
    doc = fitz.open(pdf_path)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        pix.save(
            output / f"page_{i + 1:04d}.png"
        )


render_pdf(
    "pdf/25.3.3632Dl0001-1.pdf",
    "processed_pdf/example_1"
)