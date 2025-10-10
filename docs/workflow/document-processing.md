# Document Processing

The first step in the Docs2Synth workflow is processing raw documents into structured, machine-readable formats.

## Overview

Document processing converts various document formats (PDFs, images, scanned documents) into text and layout-aware representations that can be used for downstream tasks like QA generation and retrieval.

## Supported Methods

### MinerU

MinerU is the primary OCR method for extracting text and layout from documents.

**Features:**
- High-quality text extraction
- Layout preservation
- Support for complex documents (tables, multi-column layouts)

**Usage:**

```python
from Docs2Synth.preprocess import mineru

# Process a single document
document = mineru.process_document("path/to/document.pdf")

# Process a directory of documents
documents = mineru.process_directory("path/to/documents/")
```

### Other OCR Methods

Docs2Synth supports integration with various OCR engines:

- **Tesseract**: Open-source OCR engine
- **Google Cloud Vision API**: Cloud-based OCR
- **Amazon Textract**: AWS OCR service
- **Azure Form Recognizer**: Microsoft OCR

**Generic OCR Interface:**

```python
from Docs2Synth.preprocess import ocr

# Using Tesseract
document = ocr.process_document(
    "path/to/document.pdf",
    engine="tesseract"
)

# Using cloud services
document = ocr.process_document(
    "path/to/document.pdf",
    engine="google_vision",
    api_key="your-api-key"
)
```

## Document Structure

Processed documents are returned as structured objects containing:

```python
{
    "text": "Full extracted text",
    "layout": {
        "pages": [...],
        "blocks": [...],
        "lines": [...]
    },
    "metadata": {
        "page_count": 10,
        "file_path": "path/to/document.pdf",
        "processing_time": 2.5
    }
}
```

## Best Practices

### 1. Preprocessing

Clean and normalize documents before OCR:

```python
from Docs2Synth.preprocess import preprocessing

# Enhance image quality
enhanced = preprocessing.enhance_image("document.jpg")

# Remove noise
cleaned = preprocessing.denoise(enhanced)
```

### 2. Batch Processing

For large document collections, use batch processing:

```python
from Docs2Synth.preprocess import batch

# Process documents in parallel
results = batch.process_documents(
    input_dir="documents/",
    output_dir="processed/",
    num_workers=4,
    ocr_engine="mineru"
)
```

### 3. Quality Control

Verify OCR quality before proceeding:

```python
from Docs2Synth.preprocess import quality

# Check OCR confidence
quality_score = quality.assess_ocr_quality(document)

if quality_score < 0.8:
    # Retry with different settings or manual review
    document = ocr.process_document(
        path,
        engine="google_vision"  # Try more robust engine
    )
```

## Configuration

Configure processing parameters in `config.yml`:

```yaml
preprocess:
  ocr_engine: mineru
  language: eng
  dpi: 300
  enhance_images: true
  parallel_processing: true
  num_workers: 4
```

## Error Handling

Handle common processing errors:

```python
from Docs2Synth.preprocess import ocr
from Docs2Synth.preprocess.exceptions import OCRError

try:
    document = ocr.process_document("document.pdf")
except OCRError as e:
    print(f"OCR failed: {e}")
    # Fallback to alternative method
```

## Next Steps

After processing documents, proceed to [QA Generation](qa-generation.md) to create question-answer pairs from the extracted content.

## API Reference

For detailed API documentation, see the [API Reference](../api-reference.md#preprocess).
