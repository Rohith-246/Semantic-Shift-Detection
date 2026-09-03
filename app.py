import importlib.util
import os

import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

script_path = os.path.join(os.path.dirname(__file__), "Semantic Shift Detection.py")
spec = importlib.util.spec_from_file_location("semantic_shift_detection", script_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load application script: {script_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

demo = module.create_demo()
