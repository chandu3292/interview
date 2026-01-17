import asyncio
import json
import websockets
import streamlit as st

from rag import retrieve

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
TARGET_SR = 16000
ASSISTANT_SR = 24000


async def _send_audio(ws, pcm_bytes):
    # Small helper to push a short PCM buffer and request a response.
    await ws.send(json.dumps({
        "type": "input_audio_buffer.append",
        "audio": pcm_bytes.hex()
    }))
    await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
    await ws.send(json.dumps({
        "type": "response.create",
        "response": {"modalities": ["audio"]}
    }))


async def run_realtime_loop(input_queue, output_queue, index, docs):
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(URL, additional_headers=headers, max_size=None) as ws:

        # Register the retrieval tool so the model can call it when needed.
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "Use tools to fetch PDF knowledge and reply concisely in speech.",
                "tools": [{
                    "type": "function",
                    "name": "get_information",
                    "description": "Retrieve information from the PDF knowledge base",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }]
            }
        }))

        async def sender():
            batch = bytearray()
            min_bytes = int(TARGET_SR * 0.5 * 2)  # 0.5s of 16k mono int16
            while True:
                chunk = await input_queue.get()
                if chunk is None:
                    break
                batch.extend(chunk)
                if len(batch) >= min_bytes:
                    await _send_audio(ws, bytes(batch))
                    batch.clear()

            if batch:
                await _send_audio(ws, bytes(batch))

        async def receiver():
            async for msg in ws:
                event = json.loads(msg)

                if event["type"] == "input_audio_buffer.speech_started":
                    # User interrupted; drop any pending assistant audio so playback stops fast.
                    while not output_queue.empty():
                        try:
                            output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

                elif event["type"] == "response.audio.delta":
                    delta = event.get("delta")
                    if isinstance(delta, str):
                        await output_queue.put(bytes.fromhex(delta))

                elif event["type"] == "response.function_call_arguments.done":
                    call_id = event.get("call_id")
                    args_raw = event.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                    query = args.get("query", "")
                    results = retrieve(query, index, docs) if query else []
                    context_str = "\n".join(results)

                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": context_str
                        }
                    }))

                    await ws.send(json.dumps({"type": "response.create"}))

        await asyncio.gather(sender(), receiver())
