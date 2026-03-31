import fitz
from noise import is_noise

class PDFParser:
    def __init__(self, file_stream):
        self.doc = fitz.open(stream=file_stream.read(), filetype="pdf")

        self.buffer = None
        self.batch = []
        self.token_count = 0

    def stream(self):
        for page in self.doc:
            blocks = page.get_text("dict")["blocks"]

            for block in blocks:
                yield self.process_block(block)

        yield self.flush()

    def process_block(self, block):
        if block["type"] == 0:
            text = " ".join(
                " ".join(span["text"] for span in line["spans"])
                for line in block["lines"]
            ).strip()

            if is_noise(text):
                return None

            if text:
                return {"type": "text", "content": text}

        elif block["type"] == 1:
            xref = block["image"]
            base = self.doc.extract_image(xref)

            return {"type": "image", "ext": base["ext"], "size": len(base["image"])}

    def flush(self):
        return {"type": "end"}
