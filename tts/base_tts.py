import time
from threading import Event, Lock, Thread
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from collections import deque
import queue
from queue import Queue
from io import BytesIO
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    # Engines that implement synthesize() can overlap the next request
    # with playback of the current one (Edge TTS is ~7s per sentence).
    synth_workers = 1

    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        #self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        self._tts_gen = 0

    def flush_talk(self):
        self._tts_gen += 1
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}): 
        if len(msg) > 0:
            self.msgqueue.put((msg, datainfo))

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self, quit_event):
        if self.synth_workers > 1 and type(self).synthesize is not BaseTTS.synthesize:
            self._process_tts_prefetch(quit_event)
        else:
            while not quit_event.is_set():
                try:
                    msg: tuple[str, dict] = self.msgqueue.get(block=True, timeout=1)
                    self.state = State.RUNNING
                except queue.Empty:
                    continue
                self.txt_to_audio(msg)
        self.stop_tts()
        logger.info('ttsreal thread stop')

    def _enqueue_prefetch(self, pool, pending, lock):
        with lock:
            while len(pending) < self.synth_workers:
                try:
                    msg = self.msgqueue.get(block=False)
                except queue.Empty:
                    return
                self.state = State.RUNNING
                pending.append(
                    (pool.submit(self.synthesize, msg), msg, self._tts_gen)
                )

    def _process_tts_prefetch(self, quit_event):
        """Stream the first sentence; synthesize the next ones in the background.

        Edge TTS takes several seconds per request. Waiting for the full first
        clip before any PCM is why the avatar stayed silent for ~7s. Later
        sentences still prefetch so playback does not stall between them.
        """
        pending = deque()
        lock = Lock()
        pool = ThreadPoolExecutor(max_workers=self.synth_workers)
        try:
            while not quit_event.is_set():
                if pending:
                    self._enqueue_prefetch(pool, pending, lock)
                    with lock:
                        fut, msg, gen = pending[0]
                    try:
                        pcm = fut.result(timeout=0.2)
                    except FuturesTimeout:
                        continue
                    with lock:
                        pending.popleft()
                    if gen != self._tts_gen or self.state != State.RUNNING:
                        continue
                    if pcm is None or getattr(pcm, "size", 0) == 0:
                        continue
                    self.push_pcm(pcm, msg)
                    continue

                try:
                    msg = self.msgqueue.get(block=True, timeout=0.2)
                except queue.Empty:
                    continue
                self.state = State.RUNNING
                gen = self._tts_gen
                stop = Event()

                def _fill():
                    while not stop.is_set() and not quit_event.is_set():
                        self._enqueue_prefetch(pool, pending, lock)
                        time.sleep(0.05)

                filler = Thread(target=_fill, daemon=True)
                filler.start()
                try:
                    if gen == self._tts_gen and self.state == State.RUNNING:
                        self.txt_to_audio(msg)
                finally:
                    stop.set()
                    filler.join(timeout=2)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    def synthesize(self, msg: tuple[str, dict]):
        """Return 16 kHz float32 PCM, or None. Override to enable prefetch."""
        return None

    def push_pcm(self, stream, msg: tuple[str, dict]):
        text, textevent = msg
        streamlen = stream.shape[0]
        idx = 0
        while streamlen >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            streamlen -= self.chunk
            if idx == 0:
                eventpoint = {"status": "start", "text": text}
            elif streamlen < self.chunk:
                eventpoint = {"status": "end", "text": text}
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx:idx + self.chunk], eventpoint)
            idx += self.chunk
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        pcm = self.synthesize(msg)
        if pcm is not None:
            self.push_pcm(pcm, msg)

    def stop_tts(self):
        pass
