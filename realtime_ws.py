import asyncio
import json
import base64
import websockets
import streamlit as st
from rag import retrieve

OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# The API expects 24kHz for optimal performance, though we can send what we have.
# Make sure your app.py sends raw PCM16.


async def run_realtime_loop(input_queue, output_queue, index, docs):
    headers = {
        "Authorization": f"Bearer {OPENAI_KEY}",
        "OpenAI-Beta": "realtime=v1"
    }

    async with websockets.connect(URL, additional_headers=headers) as ws:

        # 1. Initialize Session with VAD (Voice Activity Detection)
        # This tells the model to listen and decide when to speak automatically.
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500  # Wait 500ms silence before replying
                },
                "instructions": "You are a helpful assistant. Use tools to retrieve information from the PDF.",
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
            while True:
                # Get audio chunks from the microphone
                chunk = await input_queue.get()
                if chunk is None:
                    break

                # 2. Encode to Base64 (Required by OpenAI)
                b64_audio = base64.b64encode(chunk).decode("utf-8")

                # 3. Stream audio ONLY. Do NOT commit manually.
                # The Server VAD will handle the commit when you stop speaking.
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64_audio
                }))

        async def receiver():
            async for msg in ws:
                event = json.loads(msg)
                event_type = event.get("type")

                # Log errors to help debugging
                if event_type == "error":
                    print(f"Error: {event}")

                # 4. Handle Interruption
                # When the user speaks, the server sends 'speech_started'.
                # We must clear the audio queue so the bot shuts up immediately.
                elif event_type == "input_audio_buffer.speech_started":
                    print("User started speaking - clearing queue")
                    # Clear Streamlit's output queue to stop old audio
                    while not output_queue.empty():
                        try:
                            output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    # Send a cancel event to OpenAI to stop it from generating more audio
                    await ws.send(json.dumps({"type": "response.cancel"}))

                # Receive audio deltas
                elif event_type == "response.audio.delta":
                    delta = event.get("delta")
                    if delta:
                        # Decode base64 audio from server
                        audio_bytes = base64.b64decode(delta)
                        await output_queue.put(audio_bytes)

                # Handle Function Calling (RAG)
                elif event_type == "response.function_call_arguments.done":
                    call_id = event.get("call_id")
                    args_raw = event.get("arguments", "{}")
                    print(f"Calling tool: {args_raw}")

                    try:
                        args = json.loads(args_raw)
                    except:
                        args = {}

                    query = args.get("query", "")
                    results = retrieve(query, index, docs) if query else []
                    context_str = "\n".join(results)

                    # Send the tool output back
                    await ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": context_str
                        }
                    }))

                    # Trigger a response after providing tool output
                    await ws.send(json.dumps({"type": "response.create"}))

        await asyncio.gather(sender(), receiver())
