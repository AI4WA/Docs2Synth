import logging

from docs2synth.preprocess.docling_processor import _DOCLING_AVAILABLE
from docs2synth.utils.logging import setup_cli_logging

# 初始化日志（查看检测过程）
setup_cli_logging(verbose=2)

print(f"\n=== Final Check: _DOCLING_AVAILABLE = {_DOCLING_AVAILABLE} ===")
if _DOCLING_AVAILABLE:
    print("✅ Docling is detected correctly!")
else:
    print("❌ Docling is NOT detected. Check logs above.")
