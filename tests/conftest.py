"""pytest config: make repo root importable so `import tb_inference` works."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
