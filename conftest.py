# conftest.py
import sys
from pathlib import Path

# Ensure the project root is on sys.path so 'src' is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))