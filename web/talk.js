/**
 * End-user talk page: full-window WebRTC + hands-free VAD ASR → POST /human chat.
 */
(function () {
    const params = new URLSearchParams(window.location.search);

    // Energy VAD defaults (close-talk Chromium); tune in one place
    const VAD = {
        startLevel: 8,
        endLevel: 6,
        startMs: 150,
        endMs: 800,
        minSpeechMs: 400,
        maxSpeechMs: 15000,
    };

    const els = {
        video: document.getElementById("video"),
        audio: document.getElementById("audio"),
        sessionid: document.getElementById("sessionid"),
        status: document.getElementById("status"),
        transcript: document.getElementById("transcript"),
        micBtn: document.getElementById("micBtn"),
        hint: document.getElementById("hint"),
        level: document.getElementById("level"),
        connDot: document.getElementById("connDot"),
        connText: document.getElementById("connText"),
        iconMic: document.getElementById("iconMic"),
        iconMute: document.getElementById("iconMute"),
    };

    let pc = null;
    let rec = null;
    let ws = null;
    let sampleBuf = new Int16Array();
    /** @type {'connecting'|'listening'|'inSpeech'|'thinking'|'speaking'|'muted'|'idle'} */
    let phase = "connecting";
    let muted = false;
    let micRunning = false;
    let micStarting = false;
    let streamToAsr = false;
    let speechOpenPending = false;
    let awaitingAsrResult = false;

    let loudMs = 0;
    let quietMs = 0;
    let speechStartedAt = 0;
    let lastProcessAt = 0;
    let lastLevelUiAt = 0;

    function asrUrl() {
        const proto = window.location.protocol === "https:" ? "wss://" : "ws://";
        return proto + window.location.host + "/api/asr";
    }

    function setStatus(text, isError) {
        els.status.textContent = text || "";
        els.status.classList.toggle("error", !!isError);
    }

    function setLevel(powerLevel) {
        if (!els.level) return;
        const now = Date.now();
        if (now - lastLevelUiAt < 100) return;
        lastLevelUiAt = now;
        const n = Math.max(0, Math.min(100, Math.round(powerLevel || 0)));
        els.level.textContent = "lvl " + n + " (start≥" + VAD.startLevel + ")";
    }

    function setConn(ok, label) {
        els.connDot.classList.toggle("on", !!ok);
        els.connText.textContent = label || (ok ? "Connected" : "Disconnected");
    }

    function updateMicChrome() {
        const hearing = phase === "inSpeech";
        const busy = phase === "thinking" || phase === "speaking";
        els.micBtn.classList.toggle("listening", hearing && !muted);
        els.micBtn.classList.toggle("busy", busy);
        els.micBtn.classList.toggle("muted", muted);
        els.micBtn.disabled = !els.sessionid.value;
        if (els.iconMic) els.iconMic.classList.toggle("is-hidden", muted);
        if (els.iconMute) els.iconMute.classList.toggle("is-hidden", !muted);
        els.micBtn.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
        els.micBtn.title = muted ? "Unmute" : "Mute";
    }

    function setPhase(next) {
        phase = next;
        updateMicChrome();
        if (muted && next !== "connecting") {
            setStatus("Muted");
            els.hint.textContent = "Tap to unmute and keep talking hands-free";
            if (els.level) els.level.textContent = "";
            return;
        }
        if (next === "listening") {
            setStatus("Listening…");
            els.hint.textContent = "Just speak — tap mic to mute";
        } else if (next === "inSpeech") {
            setStatus("Hearing you…");
            els.hint.textContent = "Pause when finished";
        } else if (next === "thinking") {
            setStatus("Recognizing…");
            els.hint.textContent = "Please wait";
        } else if (next === "speaking") {
            setStatus("Avatar speaking…");
            els.hint.textContent = "Mic pauses until the avatar finishes";
            if (els.level) els.level.textContent = "";
        } else if (next === "connecting") {
            setStatus("Connecting to avatar…");
            els.hint.textContent = "Hands-free after connect";
        } else {
            setStatus("");
            els.hint.textContent = "Tap mic to mute or unmute";
        }
    }

    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    async function isSpeaking() {
        const sid = els.sessionid.value;
        if (!sid) return false;
        try {
            const res = await fetch("/is_speaking", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sessionid: String(sid) }),
            });
            const data = await res.json();
            return !!(data && data.data);
        } catch (e) {
            console.warn("is_speaking failed", e);
            return false;
        }
    }

    function closeWs() {
        if (ws) {
            try { ws.close(); } catch (_) {}
            ws = null;
        }
        streamToAsr = false;
        speechOpenPending = false;
        awaitingAsrResult = false;
    }

    function appendPcm16(buffer, bufferSampleRate) {
        const data_48k = buffer[buffer.length - 1];
        if (!data_48k || !data_48k.length) return;
        const data_16k = Recorder.SampleData([data_48k], bufferSampleRate, 16000).data;
        sampleBuf = Int16Array.from([...sampleBuf, ...data_16k]);
    }

    function flushPcmChunks() {
        if (!ws || ws.readyState !== 1 || sampleBuf.length === 0) return;
        const chunkSize = 960;
        while (sampleBuf.length >= chunkSize) {
            const sendBuf = sampleBuf.slice(0, chunkSize);
            sampleBuf = sampleBuf.slice(chunkSize);
            ws.send(sendBuf);
        }
        if (sampleBuf.length > 0) {
            ws.send(sampleBuf);
            sampleBuf = new Int16Array();
        }
    }

    function openAsrSession() {
        return new Promise((resolve, reject) => {
            // Keep sampleBuf — do not clear buffered speech collected during connect.
            if (ws) {
                try { ws.close(); } catch (_) {}
                ws = null;
            }
            streamToAsr = false;
            awaitingAsrResult = false;

            const url = asrUrl();
            console.log("[talk] ASR connecting", url, "bufferedSamples=", sampleBuf.length);
            setStatus("ASR connecting…");
            const socket = new WebSocket(url);
            ws = socket;
            let opened = false;
            let settled = false;

            const failTimer = setTimeout(() => {
                if (!opened) {
                    settled = true;
                    reject(new Error("ASR connect timeout: " + url));
                    closeWs();
                }
            }, 8000);

            socket.onopen = () => {
                opened = true;
                clearTimeout(failTimer);
                socket.send(JSON.stringify({
                    chunk_size: [5, 10, 5],
                    wav_name: "h5",
                    is_speaking: true,
                    chunk_interval: 10,
                    itn: false,
                    mode: "2pass",
                }));
                streamToAsr = true;
                flushPcmChunks();
                console.log("[talk] ASR open flushed PCM, remaining=", sampleBuf.length);
                if (!settled) {
                    settled = true;
                    resolve(socket);
                }
            };
            socket.onerror = () => {
                clearTimeout(failTimer);
                if (!opened && !settled) {
                    settled = true;
                    reject(new Error("ASR WebSocket error: " + url));
                }
            };
            socket.onclose = () => {
                if (ws === socket) ws = null;
                streamToAsr = false;
                if (!opened && !settled) {
                    settled = true;
                    clearTimeout(failTimer);
                    reject(new Error("ASR WebSocket closed: " + url + " (is --ASR_SERVER set?)"));
                } else if (awaitingAsrResult) {
                    awaitingAsrResult = false;
                    setStatus("ASR closed before result: " + url, true);
                    resumeListeningAfterUtterance();
                }
            };
            socket.onmessage = (evt) => {
                let msg;
                try {
                    msg = JSON.parse(evt.data);
                } catch (_) {
                    return;
                }
                const mode = msg.mode || "";
                const text = msg.text || "";
                if (mode === "2pass-offline" || mode === "offline") {
                    awaitingAsrResult = false;
                    console.log("[talk] ASR result", text.slice(0, 80));
                    setPhase("thinking");
                    sendChat(text);
                    setTimeout(() => {
                        try { socket.close(); } catch (_) {}
                    }, 200);
                }
            };
        });
    }

    function resumeListeningAfterUtterance() {
        loudMs = 0;
        quietMs = 0;
        speechStartedAt = 0;
        sampleBuf = new Int16Array();
        streamToAsr = false;
        speechOpenPending = false;
        if (muted) {
            setPhase("muted");
            return;
        }
        if (!micRunning) {
            startMicMonitor();
            return;
        }
        setPhase("listening");
    }

    function endUtterance() {
        if (phase !== "inSpeech" && !streamToAsr && !speechOpenPending) return;
        const bytes = sampleBuf.byteLength;
        console.log("[talk] utterance end bufferedBytes=", bytes);
        speechOpenPending = false;
        streamToAsr = false;
        flushPcmChunks();
        if (ws && ws.readyState === 1) {
            awaitingAsrResult = true;
            ws.send(JSON.stringify({
                chunk_size: [5, 10, 5],
                wav_name: "h5",
                is_speaking: false,
                chunk_interval: 10,
                mode: "2pass",
            }));
            setPhase("thinking");
            setStatus("Recognizing…");
        } else {
            setStatus("ASR not connected — no audio sent (" + asrUrl() + ")", true);
            resumeListeningAfterUtterance();
        }
        loudMs = 0;
        quietMs = 0;
        speechStartedAt = 0;
        // Keep Recorder running; only pause VAD via phase.
    }

    async function beginUtterance() {
        if (speechOpenPending || streamToAsr || muted) return;
        if (phase !== "listening") return;
        speechOpenPending = true;
        speechStartedAt = Date.now();
        quietMs = 0;
        console.log("[talk] utterance start bufferedSamples=", sampleBuf.length);
        try {
            await openAsrSession();
            speechOpenPending = false;
            setPhase("inSpeech");
        } catch (e) {
            console.error(e);
            speechOpenPending = false;
            streamToAsr = false;
            closeWs();
            setStatus(e.message || String(e), true);
            els.hint.textContent = "Check --ASR_SERVER and /api/asr";
            if (!muted) setPhase("listening");
        }
    }

    function onRecProcess(buffer, powerLevel, bufferDuration, bufferSampleRate) {
        if (!micRunning || muted) return;
        if (phase !== "listening" && phase !== "inSpeech") return;

        const now = Date.now();
        const dt = lastProcessAt ? Math.min(80, now - lastProcessAt) : 50;
        lastProcessAt = now;
        setLevel(powerLevel);

        // Buffer PCM as soon as we leave idle listening into speech (pending or inSpeech).
        const capturing = speechOpenPending || streamToAsr || phase === "inSpeech";
        if (capturing) {
            appendPcm16(buffer, bufferSampleRate);
            if (streamToAsr && ws && ws.readyState === 1) {
                flushPcmChunks();
            }
        }

        if (phase === "listening" && !speechOpenPending) {
            if (powerLevel >= VAD.startLevel) {
                loudMs += dt;
                quietMs = 0;
                if (loudMs >= VAD.startMs) {
                    // Start buffering this frame before WS opens
                    appendPcm16(buffer, bufferSampleRate);
                    beginUtterance();
                }
            } else {
                loudMs = 0;
            }
            return;
        }

        if (speechOpenPending && phase === "listening") {
            // Waiting for WS; keep buffering (already done above).
            return;
        }

        // inSpeech: watch for silence / max length
        const spokenMs = speechStartedAt ? now - speechStartedAt : 0;
        if (spokenMs >= VAD.maxSpeechMs) {
            endUtterance();
            return;
        }
        if (powerLevel < VAD.endLevel) {
            quietMs += dt;
            if (spokenMs >= VAD.minSpeechMs && quietMs >= VAD.endMs) {
                endUtterance();
            }
        } else {
            quietMs = 0;
        }
    }

    function ensureRecorder() {
        if (rec) return Promise.resolve(rec);
        return new Promise((resolve, reject) => {
            if (typeof Recorder === "undefined") {
                reject(new Error("Recorder not loaded"));
                return;
            }
            rec = Recorder({
                type: "pcm",
                bitRate: 16,
                sampleRate: 16000,
                onProcess: onRecProcess,
            });
            rec.open(
                () => resolve(rec),
                (err) => {
                    rec = null;
                    reject(new Error(err || "Microphone permission denied"));
                }
            );
        });
    }

    function stopMicMonitor() {
        micRunning = false;
        lastProcessAt = 0;
        if (rec) {
            try {
                rec.stop(() => {}, () => {});
            } catch (_) {}
        }
    }

    async function startMicMonitor() {
        if (muted || !els.sessionid.value || micRunning || micStarting) return;
        if (phase === "thinking" || phase === "speaking" || phase === "inSpeech") return;
        micStarting = true;
        try {
            await ensureRecorder();
            loudMs = 0;
            quietMs = 0;
            speechStartedAt = 0;
            sampleBuf = new Int16Array();
            lastProcessAt = 0;
            micRunning = true;
            rec.start();
            setPhase("listening");
        } catch (e) {
            console.error(e);
            micRunning = false;
            rec = null;
            setPhase("idle");
            setStatus("Tap the mic to allow the microphone", true);
            els.hint.textContent = e.message || String(e);
            els.micBtn.disabled = false;
        } finally {
            micStarting = false;
        }
    }

    function onAvatarReady() {
        if (!els.sessionid.value) return;
        updateMicChrome();
        if (!muted) startMicMonitor();
    }

    async function waitSpeakingEnd() {
        setPhase("speaking");
        // Do not stop Recorder permanently; pause VAD via phase only.
        for (let i = 0; i < 15; i++) {
            if (await isSpeaking()) break;
            await sleep(400);
        }
        while (await isSpeaking()) {
            await sleep(500);
        }
        await sleep(800);
        if (muted) {
            setPhase("muted");
            return;
        }
        resumeListeningAfterUtterance();
    }

    function sendChat(text) {
        const sid = els.sessionid.value;
        const cleaned = (text || "").replace(/ +/g, "").trim();
        if (!cleaned || !sid) {
            if (!cleaned) setStatus("No speech detected — listening again");
            resumeListeningAfterUtterance();
            return;
        }
        els.transcript.textContent = cleaned;
        fetch("/human", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: cleaned,
                type: "chat",
                interrupt: true,
                sessionid: String(sid),
            }),
        }).catch((e) => console.error("chat failed", e));
        waitSpeakingEnd();
    }

    function setMuted(next) {
        muted = !!next;
        if (muted) {
            if (phase === "inSpeech" || speechOpenPending) {
                streamToAsr = false;
                closeWs();
            }
            stopMicMonitor();
            setPhase("muted");
        } else {
            if (!els.sessionid.value) {
                setStatus("Wait until the avatar is connected", true);
                updateMicChrome();
                return;
            }
            if (phase === "thinking" || phase === "speaking") {
                updateMicChrome();
                setStatus(phase === "speaking" ? "Avatar speaking…" : "Recognizing…");
                els.hint.textContent = "Will listen when ready";
                return;
            }
            startMicMonitor();
        }
    }

    els.micBtn.addEventListener("click", () => {
        if (!els.sessionid.value) return;
        if (muted) {
            setMuted(false);
            return;
        }
        if (!micRunning && phase !== "inSpeech" && phase !== "thinking" && phase !== "speaking") {
            startMicMonitor();
            return;
        }
        setMuted(true);
    });

    // ── WebRTC ────────────────────────────────────────────────────
    function negotiate() {
        pc.addTransceiver("video", { direction: "recvonly" });
        pc.addTransceiver("audio", { direction: "recvonly" });
        return pc.createOffer()
            .then((offer) => pc.setLocalDescription(offer))
            .then(() => new Promise((resolve) => {
                if (pc.iceGatheringState === "complete") resolve();
                else {
                    const check = () => {
                        if (pc.iceGatheringState === "complete") {
                            pc.removeEventListener("icegatheringstatechange", check);
                            resolve();
                        }
                    };
                    pc.addEventListener("icegatheringstatechange", check);
                }
            }))
            .then(() => {
                const offer = pc.localDescription;
                const body = {
                    sdp: offer.sdp,
                    type: offer.type,
                };
                const avatar = params.get("avatar");
                const refaudio = params.get("refaudio");
                const reftext = params.get("reftext");
                if (avatar) body.avatar = avatar;
                if (refaudio) body.refaudio = refaudio;
                if (reftext) body.reftext = reftext;
                return fetch("/offer", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
            })
            .then((res) => res.json())
            .then((answer) => {
                if (answer.code && answer.code !== 0) {
                    throw new Error(answer.msg || "Server error");
                }
                if (!answer.sdp) throw new Error("Server returned no SDP");
                if (answer.sessionid) {
                    els.sessionid.value = String(answer.sessionid);
                }
                return pc.setRemoteDescription(
                    new RTCSessionDescription({ type: "answer", sdp: answer.sdp })
                ).then(() => {
                    onAvatarReady();
                });
            });
    }

    function startRtc() {
        setConn(false, "Connecting…");
        setPhase("connecting");
        const config = {
            sdpSemantics: "unified-plan",
            iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
        };
        pc = new RTCPeerConnection(config);
        pc.addEventListener("track", (evt) => {
            if (evt.track.kind === "video") {
                els.video.srcObject = evt.streams[0];
                onAvatarReady();
            } else {
                els.audio.srcObject = evt.streams[0];
            }
        });
        pc.addEventListener("connectionstatechange", () => {
            const st = pc.connectionState;
            if (st === "connected") {
                setConn(true, "Connected");
                updateMicChrome();
            } else if (st === "failed" || st === "closed") {
                setConn(false, st === "failed" ? "Failed" : "Disconnected");
                stopMicMonitor();
                closeWs();
            } else if (st === "disconnected") {
                // Transient ICE blip — keep mic / ASR running.
                setConn(false, "Reconnecting…");
            }
        });
        negotiate().catch((e) => {
            console.error(e);
            setConn(false, "Failed");
            setStatus(e.message || String(e), true);
            if (pc) {
                pc.close();
                pc = null;
            }
        });
    }

    window.addEventListener("beforeunload", () => {
        closeWs();
        stopMicMonitor();
        if (pc) {
            try { pc.close(); } catch (_) {}
        }
    });

    startRtc();
})();
