import importlib.util
import os
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Change this line to point to stream_app.py
script_path = os.path.join(os.path.dirname(__file__), "stream_app.py")

spec = importlib.util.spec_from_file_location("stream_app", script_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load application script: {script_path}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
demo = module.create_demo()
