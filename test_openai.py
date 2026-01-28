import os
from openai import OpenAI
from dotenv import load_dotenv

def test_openai_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment.")
        return

    print(f"Key value representation: {repr(api_key)}")
    print(f"Length: {len(api_key)}")
    
    # Try with raw key
    print("\n--- Testing with RAW key ---")
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        print("✅ SUCCESS (Raw)")
    except Exception as e:
        print(f"❌ FAILURE (Raw): {e}")

    # Try with stripped key
    stripped_key = api_key.strip()
    if stripped_key != api_key:
        print("\n--- Testing with STRIPPED key ---")
        client_s = OpenAI(api_key=stripped_key)
        try:
            client_s.models.list()
            print("✅ SUCCESS (Stripped)")
        except Exception as e:
            print(f"❌ FAILURE (Stripped): {e}")
    else:
        print("\nKey was already stripped.")

if __name__ == "__main__":
    test_openai_key()
