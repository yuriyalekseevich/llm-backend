import sys
print("=" * 50)
print("🧪 Setup Test for Mac")
print("=" * 50)
print(f"Python: {sys.version}")
print(f"Virtual env: {'venv' in sys.prefix}")

try:
    import openai
    print("✅ openai: INSTALLED")
except ImportError:
    print("❌ openai: MISSING")

try:
    import anthropic
    print("✅ anthropic: INSTALLED")
except ImportError:
    print("❌ anthropic: MISSING")

try:
    import dotenv
    print("✅ python-dotenv: INSTALLED")
except ImportError:
    print("❌ python-dotenv: MISSING")

print("\n🚀 If you see green checks, you're ready!")
