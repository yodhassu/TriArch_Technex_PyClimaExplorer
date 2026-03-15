import os
from dotenv import load_dotenv
import traceback

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
print("OPENAI_API_KEY from env:", repr(os.environ.get("OPENAI_API_KEY", "")))

try:
    import openai
    print("openai import: SUCCESS")
    print("openai version:", openai.__version__)
except ImportError as e:
    print("openai import: FAILED")
    print("Error:", e)
