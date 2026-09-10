/**
 * End-user talk page: full-window WebRTC + push-to-talk ASR → POST /human chat.
 */
(function () {
    const params = new URLSearchParams(window.location.search);
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
    };

    let pc = null;
    let rec = null;
    let ws = null;
    let sampleBuf = new Int16Array();
    let isRec = false;
    let holding = false;
    let toggleMode = false;
    let phase = "idle"; // idle | listening | thinking | speaking
    let pointerId = null;

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

    function setPhase(next) {
        phase = next;
        els.micBtn.classList.toggle("listening", next === "listening");
        els.micBtn.classList.toggle("busy", next === "thinking" || next === "speaking");
        const canTalk = !!els.sessionid.value && (!!pc) && next !== "thinking" && next !== "speaking";
        els.micBtn.disabled = !canTalk && next !== "listening";
        if (next === "listening") {
            setStatus("Listening…");
            els.hint.textContent = "Release to send";
        } else if (next === "thinking") {
            setStatus("Recognizing…");
            els.hint.textContent = "Please wait";
        } else if (next === "speaking") {
            setStatus("Avatar speaking…");
            els.hint.textContent = "Mic pauses until the avatar finishes";
        } else {
            setStatus(els.sessionid.value ? "Hold the mic to talk" : "Connecting…");
            els.hint.textContent = "Hold to talk · release to send · tap to toggle";
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

    async function waitSpeakingEnd() {
        setPhase("speaking");
        for (let i = 0; i < 15; i++) {
            if (await isSpeaking()) break;
            await sleep(400);
        }
        while (await isSpeaking()) {
            await sleep(500);
        }
        await sleep(800);
        setPhase("idle");
    }

    function sendChat(text) {
        const sid = els.sessionid.value;
        const cleaned = (text || "").replace(/ +/g, "").trim();
        if (!cleaned || !sid) {
            setPhase("idle");
            if (!cleaned) setStatus("No speech detected — try again");
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

    function closeWs() {
        if (ws) {
            try { ws.close(); } catch (_) {}
            ws = null;
        }
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
                resolve(socket);
            };
            socket.onerror = () => {
                clearTimeout(failTimer);
                if (!opened) reject(new Error("ASR WebSocket error"));
            };
            socket.onclose = () => {
                if (ws === socket) ws = null;
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

    function flushPcm() {
        if (!ws || ws.readyState !== 1 || sampleBuf.length === 0) return;
        ws.send(sampleBuf);
        sampleBuf = new Int16Array();
    }

    function onRecProcess(buffer, powerLevel, bufferDuration, bufferSampleRate) {
        if (!isRec) return;
        const data_48k = buffer[buffer.length - 1];
        const data_16k = Recorder.SampleData([data_48k], bufferSampleRate, 16000).data;
        sampleBuf = Int16Array.from([...sampleBuf, ...data_16k]);
        const chunkSize = 960;
        while (sampleBuf.length >= chunkSize && ws && ws.readyState === 1) {
            const sendBuf = sampleBuf.slice(0, chunkSize);
            sampleBuf = sampleBuf.slice(chunkSize);
            ws.send(sendBuf);
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

    async function startListening() {
        if (phase === "listening" || phase === "thinking" || phase === "speaking") return;
        if (!els.sessionid.value) {
            setStatus("Wait until the avatar is connected", true);
            return;
        }
        try {
            await ensureRecorder();
            await openAsrSession();
            isRec = true;
            sampleBuf = new Int16Array();
            rec.start();
            setPhase("listening");
        } catch (e) {
            console.error(e);
            isRec = false;
            closeWs();
            setPhase("idle");
            setStatus(e.message || String(e), true);
        }
    }

    function stopListening() {
        if (!isRec && phase !== "listening") return;
        isRec = false;
        holding = false;
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
        if (rec) {
            try {
                rec.stop(() => {}, () => {});
            } catch (_) {}
        }
        setPhase("thinking");
        setStatus("Recognizing…");
    }

    // ── Mic: hold (pointer) + tap toggle ──────────────────────────
    const mic = els.micBtn;
    let downAt = 0;

    mic.addEventListener("pointerdown", (ev) => {
        if (mic.disabled && phase !== "listening") return;
        ev.preventDefault();
        mic.setPointerCapture(ev.pointerId);
        pointerId = ev.pointerId;
        downAt = Date.now();
        if (toggleMode && phase === "listening") {
            stopListening();
            toggleMode = false;
            return;
        }
        holding = true;
        startListening();
    });

    function endHold(ev) {
        if (pointerId != null && ev.pointerId !== pointerId) return;
        pointerId = null;
        const heldMs = Date.now() - downAt;
        if (holding && heldMs < 280) {
            // Short tap → stay in listen (toggle mode)
            toggleMode = true;
            holding = false;
            els.hint.textContent = "Tap again to send";
            return;
        }
        if (holding || phase === "listening") {
            holding = false;
            toggleMode = false;
            stopListening();
        }
    }

    mic.addEventListener("pointerup", endHold);
    mic.addEventListener("pointercancel", endHold);
    mic.addEventListener("lostpointercapture", (ev) => {
        if (holding) endHold(ev);
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
        setStatus("Connecting to avatar…");
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
                setPhase("idle");
                els.micBtn.disabled = false;
            } else if (st === "failed" || st === "disconnected" || st === "closed") {
                setConn(false, st === "failed" ? "Failed" : "Disconnected");
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
        if (pc) {
            try { pc.close(); } catch (_) {}
        }
    });

    startRtc();
})();
