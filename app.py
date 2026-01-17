import asyncio
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer

from utils import read_pdf, chunk
from embeddings import embed, build_index
from realtime_ws import run_realtime_loop

st.set_page_config(page_title="Realtime Voice RAG", layout="centered")
st.title("📞 Realtime Voice RAG Call")

TARGET_SR = 16000
ASSISTANT_SR = 24000


class RealtimeProcessor(AudioProcessorBase):
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

    def recv_audio(self, frame):
        audio = frame.to_ndarray()
        sr = frame.sample_rate or TARGET_SR

        if audio.ndim == 2:
            # Average channels to mono.
            axis = 0 if audio.shape[0] <= 8 else 1
            audio = audio.mean(axis=axis)

        audio = audio.astype(np.float32)

        if sr != TARGET_SR:
            x_old = np.linspace(0, len(audio), num=len(audio), endpoint=False)
            x_new = np.linspace(0, len(audio), num=int(len(audio) * TARGET_SR / sr), endpoint=False)
            audio = np.interp(x_new, x_old, audio)

        pcm16 = np.clip(audio, -32768, 32767).astype(np.int16).tobytes()
        self.input_queue.put_nowait(pcm16)

        try:
            out_pcm = self.output_queue.get_nowait()
            out_np = np.frombuffer(out_pcm, dtype=np.int16)
            out_frame = av.AudioFrame.from_ndarray(out_np, format="s16", layout="mono")
            out_frame.sample_rate = ASSISTANT_SR
            return out_frame
        except asyncio.QueueEmpty:
            return None


if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.docs = None

if "call_task" not in st.session_state:
    st.session_state.call_task = None

pdf = st.file_uploader("Upload knowledge PDF", type=["pdf"])

if pdf:
    with st.spinner("Indexing document..."):
        text = read_pdf(pdf)
        chunks = chunk(text)
        vectors = embed(chunks)
        st.session_state.index = build_index(vectors)
        st.session_state.docs = chunks
    st.success("Document indexed! Start the call below.")

if st.session_state.index is None:
    st.info("Upload a PDF to enable the call.")
    st.stop()

st.subheader("Live call (full duplex)")
st.caption("Start the call, speak naturally, interrupt anytime. The assistant will call the PDF tool when needed.")

ctx = webrtc_streamer(
    key="realtime-rag",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=RealtimeProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

ready = ctx and ctx.state.playing and ctx.audio_processor

def _get_loop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

if ready and st.session_state.call_task is None:
    loop = _get_loop()
    st.session_state.call_task = loop.create_task(
        run_realtime_loop(
            ctx.audio_processor.input_queue,
            ctx.audio_processor.output_queue,
            st.session_state.index,
            st.session_state.docs,
        )
    )

if st.session_state.call_task and st.session_state.call_task.done():
    st.session_state.call_task = None
