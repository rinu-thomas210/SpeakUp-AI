// Global App Variables
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let isMuted = false;
let activeStream = null;
let currentUtterance = null;
let isInterviewActive = false;

// Audio Context & Visualizer variables
let audioCtx = null;
let analyser = null;
let dataArray = null;
let bufferLength = 0;
let drawVisual = null;

// Initialization
document.addEventListener("DOMContentLoaded", () => {
    loadOllamaModels();
    initializeVoices();
    drawSilentWave();
    
    // Bind Event Listeners
    document.getElementById("role-select").addEventListener("change", handleRoleChange);
    document.getElementById("refresh-models-btn").addEventListener("click", loadOllamaModels);
    document.getElementById("start-btn").addEventListener("click", startInterview);
    document.getElementById("record-btn").addEventListener("click", toggleRecording);
    document.getElementById("send-btn").addEventListener("click", sendTextAnswer);
    document.getElementById("toggle-input-mode-btn").addEventListener("click", toggleInputMode);
    document.getElementById("exit-btn").addEventListener("click", exitInterview);
    
    // TTS rate control
    const speedRange = document.getElementById("speed-range");
    const speedVal = document.getElementById("speed-val");
    speedRange.addEventListener("input", () => {
        speedVal.textContent = `${parseFloat(speedRange.value).toFixed(1)}x`;
    });
    
    // TTS mute button
    document.getElementById("toggle-voice-btn").addEventListener("click", toggleMute);
    
    // Keyboard bindings for convenience: Enter in text-area to send
    document.getElementById("text-answer").addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendTextAnswer();
        }
    });
});

// 1. OLLAMA MODELS LOADER
async function loadOllamaModels() {
    const modelSelect = document.getElementById("model-select");
    const refreshBtn = document.getElementById("refresh-models-btn");
    
    refreshBtn.classList.add("animate-spin");
    modelSelect.innerHTML = '<option value="">Searching local Ollama models...</option>';
    
    try {
        const response = await fetch("/api/models");
        const data = await response.json();
        
        modelSelect.innerHTML = "";
        if (data.models && data.models.length > 0) {
            data.models.forEach(model => {
                const option = document.createElement("option");
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
            
            // Prefer llama3.2:1b if available as default
            const hasLlama32 = data.models.some(m => m.includes("llama3.2:1b"));
            if (hasLlama32) {
                modelSelect.value = "llama3.2:1b";
            }
        } else {
            modelSelect.innerHTML = '<option value="llama3.2:1b">llama3.2:1b (Not found, fallback default)</option>';
        }
    } catch (err) {
        console.error("Failed to load local models:", err);
        modelSelect.innerHTML = '<option value="llama3.2:1b">llama3.2:1b (Server offline, fallback)</option>';
    } finally {
        refreshBtn.classList.remove("animate-spin");
    }
}

// Handle role dropdown change (show custom field if needed)
function handleRoleChange(e) {
    const customContainer = document.getElementById("custom-role-container");
    if (e.target.value === "custom") {
        customContainer.classList.remove("hidden");
    } else {
        customContainer.classList.add("hidden");
    }
}

// 2. TEXT-TO-SPEECH (TTS) HANDLERS
function initializeVoices() {
    const voiceSelect = document.getElementById("voice-select");
    
    function populateVoices() {
        const voices = window.speechSynthesis.getVoices();
        voiceSelect.innerHTML = '<option value="default">Default System Voice</option>';
        
        voices.forEach(voice => {
            if (voice.lang.startsWith("en-")) {
                const option = document.createElement("option");
                option.value = voice.voiceURI;
                option.textContent = `${voice.name} (${voice.lang})`;
                voiceSelect.appendChild(option);
            }
        });
    }

    populateVoices();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = populateVoices;
    }
}

function speakText(text) {
    window.speechSynthesis.cancel(); // Abort active speaking
    
    if (!isInterviewActive || isMuted) return;
    
    const statusText = document.getElementById("interviewer-status-text");
    const waveAnim = document.getElementById("audio-wave-animation");
    
    statusText.textContent = "Speaking";
    statusText.className = "status-badge speaking";
    waveAnim.classList.remove("hidden");
    
    // Remove emojis or special characters if any for cleaner TTS reading
    const cleanText = text.replace(/[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD00-\uDFFF]/g, "");
    
    const utterance = new SpeechSynthesisUtterance(cleanText);
    
    const voiceSelect = document.getElementById("voice-select");
    if (voiceSelect.value !== "default") {
        const selectedVoice = window.speechSynthesis.getVoices().find(v => v.voiceURI === voiceSelect.value);
        if (selectedVoice) {
            utterance.voice = selectedVoice;
        }
    }
    
    const speedRange = document.getElementById("speed-range");
    utterance.rate = parseFloat(speedRange.value) || 1.0;
    
    utterance.onend = () => {
        statusText.textContent = "Idle";
        statusText.className = "status-badge idle";
        waveAnim.classList.add("hidden");
    };
    
    utterance.onerror = (e) => {
        console.error("TTS synthesis error:", e);
        statusText.textContent = "Idle";
        statusText.className = "status-badge idle";
        waveAnim.classList.add("hidden");
    };
    
    currentUtterance = utterance;
    window.speechSynthesis.speak(utterance);
}

function toggleMute() {
    isMuted = !isMuted;
    const btn = document.getElementById("toggle-voice-btn");
    
    if (isMuted) {
        window.speechSynthesis.cancel();
        btn.textContent = "🔊 Unmute Voice Output";
        btn.classList.add("btn-secondary");
        btn.classList.remove("btn-primary");
        
        // Reset status if it was speaking
        const statusText = document.getElementById("interviewer-status-text");
        statusText.textContent = "Idle";
        statusText.className = "status-badge idle";
        document.getElementById("audio-wave-animation").classList.add("hidden");
    } else {
        btn.textContent = "🔇 Mute Voice Output";
        btn.classList.remove("btn-secondary");
        btn.classList.add("btn-primary");
    }
}

// 3. START INTERVIEW FLOW
async function startInterview() {
    const roleSelect = document.getElementById("role-select");
    let jobRole = roleSelect.value;
    
    if (jobRole === "custom") {
        const customInput = document.getElementById("custom-role-input");
        jobRole = customInput.value.trim();
        if (!jobRole) {
            alert("Please specify a custom job role.");
            customInput.focus();
            return;
        }
    }
    
    const modelSelect = document.getElementById("model-select");
    const modelName = modelSelect.value;
    if (!modelName) {
        alert("Please select a local Ollama model to run the interview.");
        return;
    }
    
    // Update UI Loading State
    const startBtn = document.getElementById("start-btn");
    startBtn.disabled = true;
    startBtn.textContent = "Connecting local AI...";
    
    try {
        const response = await fetch("/api/start", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ job_role: jobRole, model_name: modelName })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Server error starting interview.");
        }
        
        const data = await response.json();
        
        // Success: transition UI screens
        isInterviewActive = true;
        document.getElementById("setup-card").classList.remove("active");
        document.getElementById("setup-card").classList.add("hidden");
        document.getElementById("interview-dashboard").classList.remove("hidden");
        
        // Load first question
        const chatHistory = document.getElementById("chat-history");
        chatHistory.innerHTML = ""; // Clear placeholder
        
        appendMessage("assistant", data.question);
        speakText(data.question);
        
    } catch (err) {
        alert(`Failed to start mock interview:\n${err.message}`);
        console.error(err);
    } finally {
        startBtn.disabled = false;
        startBtn.textContent = "Start Mock Interview";
    }
}

// 4. CHAT RENDERING
function appendMessage(role, text) {
    const chatHistory = document.getElementById("chat-history");
    
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;
    
    const meta = document.createElement("div");
    meta.className = "msg-meta";
    if (role === "assistant") {
        meta.textContent = "Interviewer";
    } else if (role === "user") {
        meta.textContent = "You";
    } else if (role === "system") {
        meta.textContent = "System Alert";
    } else {
        meta.textContent = role;
    }
    
    const msgText = document.createElement("div");
    msgText.className = "msg-text";
    msgText.textContent = text;
    
    bubble.appendChild(meta);
    bubble.appendChild(msgText);
    
    chatHistory.appendChild(bubble);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// 5. AUDIO VISUALIZER DRAW LOGIC
function drawSilentWave() {
    const canvas = document.getElementById("visualizer-canvas");
    const canvasCtx = canvas.getContext("2d");
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;
    
    canvasCtx.fillStyle = 'rgba(10, 15, 30, 1)';
    canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
    
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.25)'; // faint Indigo line
    
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, HEIGHT / 2);
    canvasCtx.lineTo(WIDTH, HEIGHT / 2);
    canvasCtx.stroke();
}

function startAudioContext(stream) {
    const canvas = document.getElementById("visualizer-canvas");
    const canvasCtx = canvas.getContext("2d");
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;
    
    // audioCtx is initialized in toggleRecording synchronously
    if (!audioCtx) return;
    
    // Disconnect old source if exists
    if (window.activeAudioSource) {
        try { window.activeAudioSource.disconnect(); } catch(e) {}
    }
    
    const source = audioCtx.createMediaStreamSource(stream);
    window.activeAudioSource = source;
    
    if (!analyser) {
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 128;
    }
    
    bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
    
    source.connect(analyser);
    
    function draw() {
        if (!isRecording) {
            cancelAnimationFrame(drawVisual);
            drawSilentWave();
            return;
        }
        
        drawVisual = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);
        
        canvasCtx.fillStyle = 'rgba(10, 15, 30, 0.4)';
        canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
        
        const barWidth = (WIDTH / bufferLength) * 1.5;
        let barHeight;
        let x = 0;
        
        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 1.5;
            
            const percent = i / bufferLength;
            const r = Math.floor(99 - percent * 50);
            const g = Math.floor(102 + percent * 100);
            const b = Math.floor(241 - percent * 30);
            
            canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
            canvasCtx.fillRect(x, HEIGHT - barHeight, barWidth - 1, barHeight);
            
            x += barWidth;
        }
    }
    
    draw();
}

// 6. RECORD AUDIO ANSWER (STT) FLOW
async function toggleRecording() {
    const recordBtn = document.getElementById("record-btn");
    const statusText = document.getElementById("interviewer-status-text");
    const helpText = document.querySelector(".voice-help-text");
    
    // Stop any active synthesis
    window.speechSynthesis.cancel();
    
    // Synchronously initialize/resume AudioContext on user gesture
    if (!audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContext();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    
    if (!isRecording) {
        // Start Recording
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            activeStream = stream;
            
            // Setup MediaRecorder
            // Use WebM format as default browser support. Firefox/Chrome support audio/webm
            let options = { mimeType: 'audio/webm' };
            if (!MediaRecorder.isTypeSupported('audio/webm')) {
                options = { mimeType: 'audio/ogg' };
            }
            
            mediaRecorder = new MediaRecorder(stream, options);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                    console.log(`[REC] Chunk received: ${e.data.size} bytes`);
                }
            };
            
            mediaRecorder.onstop = async () => {
                console.log(`[REC] Recording stopped. Total chunks: ${audioChunks.length}`);
                
                // Build the blob FIRST, before cleaning up resources
                const mimeType = mediaRecorder.mimeType;
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                console.log(`[REC] Audio blob created: ${audioBlob.size} bytes, type: ${mimeType}`);
                
                // Submit to server
                if (audioBlob.size > 0) {
                    submitAudioAnswer(audioBlob);
                } else {
                    console.warn("[REC] Empty audio blob, skipping submission.");
                    alert("Recording was empty. Please try again or switch to text input.");
                }
            };
            
            isRecording = true;
            // Use timeslice of 250ms to collect audio chunks continuously
            mediaRecorder.start(250);
            console.log("[REC] Recording started with timeslice=250ms");
            
            // Adjust UI
            recordBtn.classList.add("recording");
            statusText.textContent = "Listening";
            statusText.className = "status-badge listening";
            helpText.innerHTML = "Recording active... Click mic again to <strong>stop and submit</strong>.";
            
            // Start Visuals
            startAudioContext(stream);
            
        } catch (err) {
            alert(`Microphone access denied or audio issue:\n${err.message}`);
            console.error(err);
        }
    } else {
        // Stop Recording
        isRecording = false;
        
        // ONLY stop the MediaRecorder here.
        // Stream & AudioContext cleanup is handled inside the onstop callback
        // to prevent race conditions that produce empty blobs.
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
            mediaRecorder.stop();
        }
        
        // Adjust UI
        recordBtn.classList.remove("recording");
        statusText.textContent = "Processing";
        statusText.className = "status-badge idle";
        helpText.innerHTML = "Processing your recording...";
        drawSilentWave();
    }
}

async function submitAudioAnswer(audioBlob) {
    if (!isInterviewActive) return;
    
    const statusBar = document.getElementById("status-bar");
    const statusTextEl = document.getElementById("status-text");
    const recordBtn = document.getElementById("record-btn");
    const helpText = document.querySelector(".voice-help-text");
    
    statusBar.classList.remove("hidden");
    statusTextEl.textContent = "Transcribing voice recording...";
    recordBtn.disabled = true;
    
    console.log(`[SUBMIT] Sending audio blob to server: ${audioBlob.size} bytes`);
    
    try {
        const response = await fetch("/api/transcribe-audio", {
            method: "POST",
            headers: {
                "Content-Type": audioBlob.type
            },
            body: audioBlob
        });
        
        console.log(`[SUBMIT] Transcription server responded with status: ${response.status}`);
        
        if (!isInterviewActive) return;
        const data = await response.json();
        
        if (!isInterviewActive) return;
        if (!response.ok) {
            throw new Error(data.error || "Transcription server error.");
        }
        
        if (data.error === "blank_audio") {
            statusBar.classList.add("hidden");
            // Silence was transcribed
            appendMessage("assistant", "I couldn't hear you clearly. Could you please re-record or type your answer?");
            return;
        }
        
        // Append user transcribed speech
        appendMessage("user", data.transcription);
        console.log(`[SUBMIT] User transcription displayed: "${data.transcription}"`);
        
        // Show waiting state while Ollama processes
        statusTextEl.textContent = "AI Interviewer is evaluating your response...";
        
        const textResponse = await fetch("/api/answer-text", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: data.transcription })
        });
        
        if (!isInterviewActive) return;
        if (!textResponse.ok) {
            const errData = await textResponse.json();
            throw new Error(errData.error || "Server processing failed.");
        }
        
        const aiData = await textResponse.json();
        
        if (!isInterviewActive) return;
        statusBar.classList.add("hidden");
        
        // Append AI response & Speak
        appendMessage("assistant", aiData.response);
        console.log(`[SUBMIT] AI response displayed.`);
        speakText(aiData.response);
        
    } catch (err) {
        if (!isInterviewActive) return;
        statusBar.classList.add("hidden");
        console.error("[SUBMIT] Error:", err);
        appendMessage("assistant", `Error processing your response: ${err.message}. Please try again or switch to text input.`);
    } finally {
        if (isInterviewActive) {
            recordBtn.disabled = false;
            if (helpText) {
                helpText.innerHTML = "Click the mic to <strong>start recording</strong>. Click again when <strong>finished</strong>.";
            }
        }
    }
}

// 7. TEXT FALLBACK ANSWER FLOW
async function sendTextAnswer() {
    if (!isInterviewActive) return;
    
    const inputArea = document.getElementById("text-answer");
    const answerText = inputArea.value.trim();
    if (!answerText) return;
    
    // Stop synthesis
    window.speechSynthesis.cancel();
    
    const sendBtn = document.getElementById("send-btn");
    const statusBar = document.getElementById("status-bar");
    const statusText = document.getElementById("status-text");
    
    // Clear input
    inputArea.value = "";
    
    // Add user message to UI
    appendMessage("user", answerText);
    
    statusBar.classList.remove("hidden");
    statusText.textContent = "Interviewer is evaluating your response...";
    sendBtn.disabled = true;
    inputArea.disabled = true;
    
    try {
        const response = await fetch("/api/answer-text", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: answerText })
        });
        
        if (!isInterviewActive) return;
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Server processing failed.");
        }
        
        const data = await response.json();
        
        if (!isInterviewActive) return;
        statusBar.classList.add("hidden");
        
        appendMessage("assistant", data.response);
        speakText(data.response);
        
    } catch (err) {
        if (!isInterviewActive) return;
        statusBar.classList.add("hidden");
        alert(`Failed to submit answer:\n${err.message}`);
        console.error(err);
    } finally {
        if (isInterviewActive) {
            sendBtn.disabled = false;
            inputArea.disabled = false;
            inputArea.focus();
        }
    }
}

// Toggle Mode between Text and Voice
function toggleInputMode() {
    const voiceSec = document.getElementById("voice-input-section");
    const textSec = document.getElementById("text-input-section");
    const toggleBtn = document.getElementById("toggle-input-mode-btn");
    
    if (voiceSec.classList.contains("active")) {
        // Switch to Text
        voiceSec.classList.remove("active");
        voiceSec.classList.add("hidden");
        
        textSec.classList.remove("hidden");
        textSec.classList.add("active");
        
        toggleBtn.textContent = "🎙️ Prefer voice? Switch to Microphone Input";
        document.getElementById("text-answer").focus();
    } else {
        // Switch to Voice
        textSec.classList.remove("active");
        textSec.classList.add("hidden");
        
        voiceSec.classList.remove("hidden");
        voiceSec.classList.add("active");
        
        toggleBtn.textContent = "⌨️ Prefer typing? Switch to Text Input";
    }
}

// 8. EXIT INTERVIEW
function exitInterview() {
    if (confirm("Are you sure you want to exit the current interview? Your session history will be reset.")) {
        isInterviewActive = false;
        // Stop any speech or recording
        window.speechSynthesis.cancel();
        if (isRecording) {
            toggleRecording();
        }
        
        // Show setup screen
        document.getElementById("interview-dashboard").classList.add("hidden");
        document.getElementById("setup-card").classList.remove("hidden");
        document.getElementById("setup-card").classList.add("active");
    }
}
