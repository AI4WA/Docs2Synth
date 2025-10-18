from docs2synth.preprocess.schema import (
    DocumentMetadata,
    DocumentObject,
    DocumentProcessResult,
    LabelType,
    ProcessMetadata,
)


def test_document_object_to_dict_filters_optionals():
    obj = DocumentObject(
        object_id=0,
        text="hello",
        bbox=(1.0, 2.0, 3.0, 4.0),
        label=LabelType.TEXT,
        page=None,
        score=None,
        extra={},
    )
    d = obj.to_dict()
    assert d["text"] == "hello"
    assert d["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert d["label"] == "text"
    assert "page" not in d
    assert "score" not in d


def test_process_metadata_to_dict_filters_none():
    pm = ProcessMetadata(
        processor_name="ocr", timestamp="2025-10-15T00:00:00Z", latency=None
    )
    d = pm.to_dict()
    assert d == {"processor_name": "ocr", "timestamp": "2025-10-15T00:00:00Z"}


def test_document_metadata_to_dict_filters_none_but_keeps_extra_key():
    dm = DocumentMetadata(source="/doc.png", extra={})
    d = dm.to_dict()
    assert d["source"] == "/doc.png"
    assert "extra" in d and d["extra"] == {}
    assert "mime_type" not in d


def test_result_round_trip_minimal():
    # Build minimal result
    obj0 = DocumentObject(
        object_id=0,
        text="901016",
        bbox=(44.0, 391.0, 44.0, 410.0),
        label=LabelType.TEXT,
    )
    res = DocumentProcessResult(
        objects={0: obj0},
        object_list=[],
        bbox_list=[obj0.bbox],
        context="-TICKET CP 901016 ...",
        reading_order_ids=[0],
        process_metadata=ProcessMetadata(
            processor_name="tesseract-ocr",
            timestamp="2025-10-15T12:00:00Z",
            latency=123.4,
        ),
        document_metadata=DocumentMetadata(source="/path/file.png", page_count=1),
    )
    payload = res.to_dict()
    assert "objects" in payload and 0 in payload["objects"]
    assert payload["objects"][0]["label"] == "text"
    assert payload["bbox_list"] == [[44.0, 391.0, 44.0, 410.0]]
    assert payload["process_metadata"]["processor_name"] == "tesseract-ocr"

    # Round-trip via from_dict
    res2 = DocumentProcessResult.from_dict(payload)
    assert res2.context == res.context
    assert res2.reading_order_ids == [0]
    assert res2.process_metadata.processor_name == "tesseract-ocr"


def test_from_dict_pure_behavior():
    payload = {
        "objects": {
            "0": {"text": "A", "bbox": [0, 1, 2, 3], "label": "text"},
            1: {"text": "B", "bbox": [4, 5, 6, 7], "label": "sentence"},
        },
        "object_list": [
            {"text": "X", "bbox": [10, 11, 12, 13], "label": "paragraph"},
            {"text": "Y", "bbox": [14, 15, 16, 17], "label": "other"},
        ],
        "bbox_list": [[0, 1, 2, 3]],
        "context": "A B",
        "reading_order_ids": [1, 0],
        "process_metadata": {
            "processor_name": "unit-test",
            "timestamp": "2025-10-15T00:00:00Z",
            "latency": 1.23,
        },
        "document_metadata": {"source": "/tmp/in.png"},
    }
    res = DocumentProcessResult.from_dict(payload)

    # objects keys should be ints 0 and 1
    assert set(res.objects.keys()) == {0, 1}
    assert res.objects[0].text == "A"
    assert res.objects[1].label == LabelType.SENTENCE

    # object_list assigned by enumerate, independent of objects
    assert len(res.object_list) == 2
    assert res.object_list[0].text == "X"
    assert res.object_list[1].bbox == (14.0, 15.0, 16.0, 17.0)

    # bbox_list normalized to tuples of float
    assert res.bbox_list == [(0.0, 1.0, 2.0, 3.0)]

    # metadata loaded as-is
    assert res.process_metadata.processor_name == "unit-test"
    assert res.process_metadata.latency == 1.23
    assert res.document_metadata.source == "/tmp/in.png"
