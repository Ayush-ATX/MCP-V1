"""Quick manual integration check for Server 3's reformulation model."""
import asyncio
import os
import sys

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
        for variant in result["variants"]:
            print(f"  [{variant['kind']:10}] {variant['text']}")
    else:
        print("Error:", result.get("error"))


if __name__ == "__main__":
    asyncio.run(main())
