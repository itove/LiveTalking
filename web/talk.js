/**
 * End-user talk page: full-window WebRTC + hands-free VAD ASR → POST /human chat.
 */
(function () {
    const params = new URLSearchParams(window.location.search);

    // Energy VAD defaults (close-talk Chromium); tune in one place
    const VAD = {
        startLevel: 15,
        endLevel: 12,
        startMs: 200,
        endMs: 900,
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
    let streamToAsr = false;
    let speechOpenPending = false;

    let loudMs = 0;
    let quietMs = 0;
    let speechStartedAt = 0;
    let lastProcessAt = 0;

    function asrUrl() {
        const proto = window.location.protocol === "https:" ? "wss://" : "ws://";
        return proto + window.location.host + "/api/asr";
    }

    function setStatus(text, isError) {
        els.status.textContent = text || "";
        els.status.classList.toggle("error", !!isError);
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
        els.micBtn.disabled = phase === "connecting" || !els.sessionid.value;
        if (els.iconMic) els.iconMic.hidden = muted;
        if (els.iconMute) els.iconMute.hidden = !muted;
        els.micBtn.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
        els.micBtn.title = muted ? "Unmute" : "Mute";
    }

    function setPhase(next) {
        phase = next;
        updateMicChrome();
        if (muted && next !== "connecting") {
            setStatus("Muted");
            els.hint.textContent = "Tap to unmute and keep talking hands-free";
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
    }

    function flushPcm() {
        if (!ws || ws.readyState !== 1 || sampleBuf.length === 0) return;
        ws.send(sampleBuf);
        sampleBuf = new Int16Array();
    }

    function openAsrSession() {
        return new Promise((resolve, reject) => {
            closeWs();
            sampleBuf = new Int16Array();
            const socket = new WebSocket(asrUrl());
            ws = socket;
            let opened = false;

            const failTimer = setTimeout(() => {
                if (!opened) {
                    reject(new Error("ASR connect timeout"));
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
                resolve(socket);
            };
            socket.onerror = () => {
                clearTimeout(failTimer);
                if (!opened) reject(new Error("ASR WebSocket error"));
            };
            socket.onclose = () => {
                if (ws === socket) ws = null;
                streamToAsr = false;
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
                    setPhase("thinking");
                    sendChat(text);
                    setTimeout(() => {
                        try { socket.close(); } catch (_) {}
                    }, 200);
                }
            };
        });
    }

    function endUtterance() {
        if (phase !== "inSpeech" && !streamToAsr) return;
        streamToAsr = false;
        speechOpenPending = false;
        flushPcm();
        if (ws && ws.readyState === 1) {
            ws.send(JSON.stringify({
                chunk_size: [5, 10, 5],
                wav_name: "h5",
                is_speaking: false,
                chunk_interval: 10,
                mode: "2pass",
            }));
        }
        loudMs = 0;
        quietMs = 0;
        speechStartedAt = 0;
        setPhase("thinking");
        setStatus("Recognizing…");
        stopMicMonitor();
    }

    async function beginUtterance() {
        if (speechOpenPending || streamToAsr || muted) return;
        if (phase !== "listening") return;
        speechOpenPending = true;
        speechStartedAt = Date.now();
        quietMs = 0;
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
            // Stay listening so user can try again
            if (!muted) setPhase("listening");
        }
    }

    function onRecProcess(buffer, powerLevel, bufferDuration, bufferSampleRate) {
        if (!micRunning || muted) return;
        if (phase !== "listening" && phase !== "inSpeech") return;

        const now = Date.now();
        const dt = lastProcessAt ? Math.min(80, now - lastProcessAt) : 50;
        lastProcessAt = now;

        if (phase === "listening") {
            if (powerLevel >= VAD.startLevel) {
                loudMs += dt;
                quietMs = 0;
                if (loudMs >= VAD.startMs) beginUtterance();
            } else {
                loudMs = 0;
            }
            return;
        }

        // inSpeech: stream PCM + watch for silence / max length
        if (streamToAsr && ws && ws.readyState === 1) {
            const data_48k = buffer[buffer.length - 1];
            const data_16k = Recorder.SampleData([data_48k], bufferSampleRate, 16000).data;
            sampleBuf = Int16Array.from([...sampleBuf, ...data_16k]);
            const chunkSize = 960;
            while (sampleBuf.length >= chunkSize) {
                const sendBuf = sampleBuf.slice(0, chunkSize);
                sampleBuf = sampleBuf.slice(chunkSize);
                ws.send(sendBuf);
            }
        }

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
        if (muted || !els.sessionid.value) return;
        if (phase === "thinking" || phase === "speaking" || phase === "connecting") return;
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
            setStatus(e.message || String(e), true);
            setPhase("idle");
        }
    }

    async function waitSpeakingEnd() {
        setPhase("speaking");
        stopMicMonitor();
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
        await startMicMonitor();
    }

    function sendChat(text) {
        const sid = els.sessionid.value;
        const cleaned = (text || "").replace(/ +/g, "").trim();
        if (!cleaned || !sid) {
            if (!cleaned) setStatus("No speech detected — listening again");
            if (muted) {
                setPhase("muted");
            } else {
                startMicMonitor();
            }
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
            if (phase === "inSpeech") {
                // Cancel in-progress utterance without chatting
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
        if (els.micBtn.disabled) return;
        setMuted(!muted);
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
                );
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
            } else {
                els.audio.srcObject = evt.streams[0];
            }
        });
        pc.addEventListener("connectionstatechange", () => {
            const st = pc.connectionState;
            if (st === "connected") {
                setConn(true, "Connected");
                updateMicChrome();
                if (!muted) startMicMonitor();
            } else if (st === "failed" || st === "disconnected" || st === "closed") {
                setConn(false, st === "failed" ? "Failed" : "Disconnected");
                stopMicMonitor();
                closeWs();
                els.micBtn.disabled = true;
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
