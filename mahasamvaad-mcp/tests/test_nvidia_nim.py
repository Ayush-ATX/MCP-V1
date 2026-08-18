"""Quick integration test for Server 3 NVIDIA NIM paraphrase generation."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server_reformulation.main import generate_paraphrases


async def main():
    result = await generate_paraphrases(
        query="Who can file an RTI application?",
        n=3,
        include_marathi=True,
    )
    print("ok =", result.get("ok"))
    if result.get("ok"):
        for v in result.get("variants", []):
            kind = v["kind"]
            text = v["text"]
            print(f"  [{kind:10}] {text}")
    else:
        print("Error:", result.get("error"))


if __name__ == "__main__":
    asyncio.run(main())
