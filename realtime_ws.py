import json
import websockets
import streamlit as st

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

async def stream_response(user_text, rag_context):
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(URL, additional_headers=headers) as ws:

        # Inject RAG context FIRST
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "system",
                "content": [{
                    "type": "input_text",
                    "text": (
                        "Use ONLY the following context to answer. "
                        "If the answer is not present, say you don't know.\n\n"
                        f"{rag_context}"
                    )
                }]
            }
        }))

        # User message
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": user_text
                }]
            }
        }))

        # Trigger response
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text"]
            }
        }))

        async for msg in ws:
            event = json.loads(msg)
            print(event)  # Debug logging

            if event["type"] == "response.text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield delta

            if event["type"] == "response.done":
                break
