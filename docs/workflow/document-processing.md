# Document Processing

Extract structured text and layout from documents (PDFs, images).

## Processors

### Docling

**Best for**: Advanced layout understanding, complex documents

```bash
docs2synth preprocess document.pdf --processor docling
```

- Advanced layout analysis
- Table and figure extraction
- Multi-page support
- Reading order detection

---

### PaddleOCR

**Best for**: General OCR, multilingual documents, Asian languages

```bash
docs2synth preprocess document.png --processor paddleocr --lang en
```

- 80+ languages
- GPU acceleration
- Bounding boxes with confidence scores
- Fast and accurate

---

### PDFPlumber

**Best for**: Parsed PDFs with text layers (digital documents)

```bash
docs2synth preprocess document.pdf --processor pdfplumber
```

- Very fast (no OCR needed)
- Word-level bounding boxes
- Table extraction
- Only works with text-based PDFs

---

### EasyOCR

**Best for**: Alternative OCR, 80+ languages

```bash
docs2synth preprocess document.png --processor easyocr
```

- 80+ languages
- GPU acceleration
- Easy setup
- Good for scanned documents

---

## Usage

```bash
# Single document
docs2synth preprocess document.png

# Directory
docs2synth preprocess ./documents/

# With options
docs2synth preprocess ./documents/ \
  --processor docling \
  --output-dir ./processed \
  --lang en \
  --device gpu
```

## Output Format

JSON files with:
- `objects`: Text objects with `text`, `bbox`, `page`, `label`
- `context`: Full document text in reading order
- `reading_order_ids`: Object sequence
- `process_metadata`: Processor info, timestamp
- `document_metadata`: Filename, dimensions

**Example:**
```json
{
  "objects": {
    "obj_0": {
      "text": "Invoice Number: INV-2024-001",
      "bbox": [100, 200, 300, 220],
      "page": 0,
      "label": "text"
    }
  },
  "context": "Invoice Number: INV-2024-001\n...",
  "reading_order_ids": ["obj_0", "obj_1", ...],
  "process_metadata": {
    "processor_name": "docling",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Processor Comparison

| Processor | Speed | Languages | Tables | OCR | Best For |
|-----------|-------|-----------|--------|-----|----------|
| Docling | Medium | All | ✅ | ✅ | Complex layouts |
| PaddleOCR | Fast | 80+ | ❌ | ✅ | General OCR |
| PDFPlumber | Very Fast | All | ✅ | ❌ | Text PDFs |
| EasyOCR | Medium | 80+ | ❌ | ✅ | Alternative OCR |

## Next Steps

- [QA Generation](qa-generation.md) - Generate QA pairs
- [Retriever Training](retriever-training.md) - Train models
- [RAG Deployment](rag-path.md) - Build RAG systems

For complete workflow: [Complete Workflow Guide](complete-workflow.md)

## API Reference

See [API Reference](../api-reference.md) page.
