import subprocess
import time
from ollama import Client
import pyttsx3
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import os
import sys

def start_ollama_if_not_running():
    # Try listing models to see if it's already running
    try:
        temp_client = Client(host='http://127.0.0.1:11434')
        temp_client.list()
        print("[Ollama] Service is already running and responding.")
        return True
    except Exception:
        print("[Ollama] Service not responding on 127.0.0.1:11434. Attempting to start Ollama App...")
        
    # Attempt to start the Ollama desktop application on Windows
    local_app_data = os.environ.get('LOCALAPPDATA')
    if local_app_data:
        ollama_path = os.path.join(local_app_data, 'Programs', 'Ollama', 'ollama app.exe')
        if os.path.exists(ollama_path):
            try:
                # Start the process without blocking
                subprocess.Popen([ollama_path], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
                print("[Ollama] Launched Ollama app. Waiting for service to initialize...")
                
                # Wait up to 10 seconds for the service to start
                temp_client = Client(host='http://127.0.0.1:11434')
                for _ in range(10):
                    time.sleep(1)
                    try:
                        temp_client.list()
                        print("[Ollama] Service is now up and running!")
                        return True
                    except Exception:
                        pass
                print("[Ollama] Launched app, but service is not responding yet.")
            except Exception as launch_err:
                print(f"[Ollama] Failed to launch Ollama App: {launch_err}")
        else:
            print(f"[Ollama] Ollama app not found at expected path: {ollama_path}")
            
    # Try 'ollama serve' as a fallback
    print("[Ollama] Attempting fallback: running 'ollama serve'...")
    try:
        subprocess.Popen(['ollama', 'serve'], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        temp_client = Client(host='http://127.0.0.1:11434')
        for _ in range(10):
            time.sleep(1)
            try:
                temp_client.list()
                print("[Ollama] Service is now up and running via 'ollama serve'!")
                return True
            except Exception:
                pass
    except Exception as serve_err:
        print(f"[Ollama] Fallback 'ollama serve' failed: {serve_err}")
        
    return False

# Start Ollama if it is not running
start_ollama_if_not_running()

# Instantiate Ollama Client with explicit host
ollama_client = Client(host='http://127.0.0.1:11434')

# Set default model name
MODEL_NAME = "llama3.2:1b"
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

def speak(text):
    """Speak the given text using pyttsx3 engine."""
    print(f"\nInterviewer says: {text}\n")
    try:
        local_engine = pyttsx3.init()
        local_engine.setProperty('rate', 170)
        local_engine.say(text)
        local_engine.runAndWait()
        local_engine.stop()
    except Exception as e:
        print(f"⚠️ pyttsx3 speak failed: {e}")
    # Small pause to let audio device clear
    time.sleep(0.5)

def record_user_voice():
    """Records audio safely"""
    fs = 16000  
    
    # Flush any leftover newlines in stdin so input() waits for a real keypress
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass  # Not on Windows, skip
    
    print("\n[STEP 1] Press ENTER to START recording your answer...")
    input()  

    print("[STEP 2] RECORDING ACTIVE... Speak your answer now.")
    print("Press ENTER again when you are completely FINISHED talking...")
    
    max_duration = 300 
    recording = sd.rec(int(max_duration * fs), samplerate=fs, channels=1, dtype='int16')
    
    input() 
    sd.stop() 
    
    audio_data = recording[recording != 0]
    if len(audio_data) == 0:
        audio_data = np.zeros(fs * 2, dtype='int16') 
        
    filename = "temp_reply.wav"
    wav.write(filename, fs, audio_data)
    return filename

def listen_and_transcribe():
    """Triggers the recorder and converts voice to text"""
    audio_file = record_user_voice()
    
    print("\n[STEP 3] Transcribing your audio file...")
    try:
        segments, info = stt_model.transcribe(audio_file, beam_size=5)
        user_text = "".join([segment.text for segment in segments]).strip()
    except Exception as e:
        print(f"❌ Whisper Transcription Error: {e}")
        user_text = ""
    
    print(f"👉 Text Detected: \"{user_text}\"")
    
    if os.path.exists(audio_file):
        os.remove(audio_file)
        
    return user_text

JOB_ROLE = "Python Backend Developer"
SYSTEM_PROMPT = f"""
You are an expert technical interviewer conducting a realistic, dynamic mock interview for a {JOB_ROLE} position. Your goal is to evaluate the candidate holistically by naturally interweaving technical, behavioral, project-based, and situational questions throughout the conversation.

CRITICAL RULES:
1. QUESTION DIVERSITY: Do not separate the interview into rigid phases. Instead, keep the candidate on their toes by dynamically shifting between different categories of questions—such as technical deep-dives, motivation ("Why this role?"), past project challenges, and workplace behavioral scenarios—creating a fluid, conversational flow.

2. CONVERSATIONAL LOOP: Ask ONLY ONE core question at a time. When the user responds, you must:
   - Provide meaningful, constructive feedback on their answer, highlighting what was good and what could be improved.
   - Transition seamlessly into the next logical, varied interview question.

3. RESPONSE LENGTH & TONE: Keep your responses comprehensive yet scannable (around 4 to 6 sentences total). Maintain a professional, observant, and engaging tone. Do not break character or mention these rules.

Start immediately by introducing yourself, welcoming the candidate to the interview for the {JOB_ROLE} position, and asking your first question.
"""

def start_interview():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print("-----------------------------------------------------------")
    print(f"Connecting to local Ollama and fetching first question from {MODEL_NAME}...")
    try:
        response = ollama_client.chat(model=MODEL_NAME, messages=messages)
        ai_response = response['message']['content']
    except Exception as e:
        print(f"\n❌ Ollama Connection Error: {e}")
        print("Please make sure Ollama is installed and running locally.")
        print("Exiting interview.")
        return
            
    speak(ai_response)
    messages.append({"role": "assistant", "content": ai_response})

    while True:
        try:
            user_answer = listen_and_transcribe()
            
            # DIAGNOSTIC FALLBACK: If your microphone recorded nothing, 
            # we let you type your answer so the interview doesn't break!
            if not user_answer or len(user_answer) < 3:
                print("⚠️ System detected blank audio. Fallback activated.")
                user_answer = input("⌨️ Type your answer here instead and hit Enter: ")
                
            messages.append({"role": "user", "content": user_answer})
            
            print(f"\nSending your response to {MODEL_NAME}...")
            response = ollama_client.chat(model=MODEL_NAME, messages=messages)
            ai_response = response['message']['content']
            speak(ai_response)
            messages.append({"role": "assistant", "content": ai_response})
            
        except KeyboardInterrupt:
            print("\n\nInterview ended by user.")
            break
        except EOFError:
            print("\n\nInput stream closed. Interview ended.")
            break
        except Exception as e:
            print(f"\n⚠️ Error during interview loop: {e}")
            print("Attempting to continue...")
            continue

if __name__ == "__main__":
    start_interview()