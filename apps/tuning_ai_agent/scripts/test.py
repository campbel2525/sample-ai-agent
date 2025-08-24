import anthropic

from config.settings import Settings

client = anthropic.Anthropic(api_key=Settings().anthropic_api_key)
resp = client.messages.create(
    model=Settings().anthropic_model,
    max_tokens=50,
    messages=[{"role": "user", "content": "1文で自己紹介して"}],
)
print("".join(b.text for b in resp.content if b.type == "text"))
