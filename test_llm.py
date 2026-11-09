"""
Run:  python test_llm.py
Checks your keys and shows which AI provider answered. Prints no keys.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import llm_client as c

print("Order:", c._provider_order())
for name in c._provider_order():
    has = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}.get(name)
    if has:
        print(f"  {name}: key {'FOUND' if os.getenv(has, '').strip() else 'MISSING'} in .env")

for name in c._provider_order():
    try:
        out = c._PROVIDERS[name]("Reply with exactly the single word: OK", 20)
        print(f"[{name}] works -> {out[:60]!r}")
    except Exception as e:
        print(f"[{name}] FAILED -> {e}")
