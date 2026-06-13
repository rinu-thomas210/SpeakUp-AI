from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import json
import urllib.parse
import uuid
import subprocess
import time
from ollama import Client
from faster_whisper import WhisperModel

# Global Session State (Ollama Only)
session_state = {
    "messages": [],
    "model_name": "llama3.2:1b",
    "job_role": "Python Backend Developer"
}

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


# Load faster_whisper model on startup
print("Initializing Whisper Speech-to-Text Model (tiny.en)...")
try:
    stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    print("[OK] Whisper Model Loaded successfully!")
except Exception as e:
    print(f"[ERROR] Whisper Model Load Failed: {e}")
    stt_model = None

class InterviewRequestHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Override to suppress default request logging to avoid terminal clutter
        pass

    def send_json_response(self, status_code, data):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        # Handle preflight CORS requests gracefully
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        # API routing
        if path == "/api/models":
            self.handle_get_models()
            return
            
        # Static files serving
        if path == "/":
            path = "/index.html"
            
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        safe_path = os.path.normpath(path.lstrip("/"))
        file_path = os.path.join(static_dir, safe_path)
        
        # Security check: verify path remains within static directory
        if not file_path.startswith(static_dir) or not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_error(404, "File Not Found")
            return
            
        content_type = "text/plain"
        if file_path.endswith(".html"):
            content_type = "text/html"
        elif file_path.endswith(".css"):
            content_type = "text/css"
        elif file_path.endswith(".js"):
            content_type = "text/javascript"
        elif file_path.endswith(".png"):
            content_type = "image/png"
        elif file_path.endswith(".jpg") or file_path.endswith(".jpeg"):
            content_type = "image/jpeg"
        elif file_path.endswith(".svg"):
            content_type = "image/svg+xml"
        elif file_path.endswith(".ico"):
            content_type = "image/x-icon"
            
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Server Error: {e}")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        
        if path == "/api/start":
            self.handle_start_interview()
        elif path == "/api/answer-text":
            self.handle_answer_text()
        elif path == "/api/transcribe-audio":
            self.handle_transcribe_audio()
        else:
            self.send_error(404, "Endpoint Not Found")

    def read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def handle_get_models(self):
        models = []
        try:
            models_response = ollama_client.list()
            ollama_models = []
            if hasattr(models_response, 'models'):
                for m in models_response.models:
                    if hasattr(m, 'model'):
                        ollama_models.append(m.model)
                    elif hasattr(m, 'name'):
                        ollama_models.append(m.name)
            elif isinstance(models_response, dict) and 'models' in models_response:
                for m in models_response['models']:
                    if isinstance(m, dict) and 'model' in m:
                        ollama_models.append(m['model'])
                    elif isinstance(m, dict) and 'name' in m:
                        ollama_models.append(m['name'])
            
            # Deduplicate
            models = sorted(list(set(ollama_models)))
            self.send_json_response(200, {"models": models})
        except Exception as e:
            print(f"[WARN] Failed to list Ollama models: {e}")
            self.send_json_response(200, {"models": [], "error": str(e)})

    def handle_start_interview(self):
        try:
            data = self.read_json_body()
            job_role = data.get("job_role", "Python Backend Developer").strip()
            model_name = data.get("model_name", "llama3.2:1b").strip()
            
            session_state["job_role"] = job_role
            session_state["model_name"] = model_name
            
            system_prompt = f"""
You are an expert technical interviewer conducting a realistic, dynamic mock interview for a {job_role} position. Your goal is to evaluate the candidate holistically by naturally interweaving technical, behavioral, project-based, and situational questions throughout the conversation.

CRITICAL RULES:
1. PLAY ONLY ONE ROLE: You are the interviewer. You must NEVER simulate or write the candidate's answers. Stop generating immediately after you ask your question.

2. CONVERSATIONAL LOOP: Ask ONLY ONE question at a time. When the user provides an answer, you must:
   a. Briefly acknowledge or evaluate their response (1-2 sentences max).
   b. Then, seamlessly transition to the NEXT interview question.

3. TONE & PERSONA: Speak like a real senior interviewer—professional, observant, and engaging. Use natural transitions like "That's a great point, now switching gears..."

4. CONCISENESS: Keep your responses under 4 sentences total (feedback + next question combined). Do NOT output conversational formatting like "Answer:", "Follow-up:", or "Transition:".

5. NEVER BREAK CHARACTER. Start immediately by introducing yourself and asking the first question, then STOP and wait for the candidate to speak.
"""
            session_state["messages"] = [{"role": "system", "content": system_prompt}]
            
            print(f"[START] Starting mock interview for: {job_role} (Model: {model_name})")
            
            # Use local Ollama
            response = ollama_client.chat(model=model_name, messages=session_state["messages"])
            ai_response = response['message']['content']
            session_state["messages"].append({"role": "assistant", "content": ai_response})
            
            self.send_json_response(200, {
                "question": ai_response,
                "status": "started",
                "fallback": False
            })
        except Exception as e:
            print(f"[ERROR] Error starting interview: {e}")
            self.send_json_response(500, {"error": f"Failed to generate first question: {e}"})

    def handle_answer_text(self):
        try:
            data = self.read_json_body()
            user_answer = data.get("text", "").strip()
            
            if not user_answer:
                self.send_json_response(400, {"error": "Answer content cannot be empty"})
                return
                
            session_state["messages"].append({"role": "user", "content": user_answer})
            print(f"[INFO] Processed user typed answer. Sending to Ollama...")
            
            response = ollama_client.chat(model=session_state["model_name"], messages=session_state["messages"])
            ai_response = response['message']['content']
            print(f"[OK] Ollama responded. Sending reply to browser.")
            
            session_state["messages"].append({"role": "assistant", "content": ai_response})
            
            self.send_json_response(200, {
                "transcription": user_answer,
                "response": ai_response
            })
        except Exception as e:
            print(f"[ERROR] Error in text answer: {e}")
            self.send_json_response(500, {"error": f"Failed to generate response: {e}"})

    def handle_transcribe_audio(self):
        try:
            if not stt_model:
                self.send_json_response(500, {"error": "Whisper STT model was not initialized successfully on backend."})
                return
                
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response(400, {"error": "Empty audio payload"})
                return
                
            audio_bytes = self.rfile.read(content_length)
            
            # Generate unique filename for temp file
            temp_filename = f"temp_web_{uuid.uuid4().hex}.webm"
            
            with open(temp_filename, "wb") as f:
                f.write(audio_bytes)
                
            print(f"[INFO] Received audio, transcribing {temp_filename}...")
            
            try:
                segments, info = stt_model.transcribe(temp_filename, beam_size=5)
                user_text = "".join([segment.text for segment in segments]).strip()
            except Exception as e:
                print(f"[ERROR] Whisper transcription failed: {e}")
                self.send_json_response(500, {"error": f"Whisper speech-to-text failed: {e}"})
                return
            finally:
                # Cleanup audio file immediately
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception as clean_err:
                        print(f"[WARN] Could not remove temp audio file: {clean_err}")
            
            print(f"[INFO] Transcribed: {user_text}")
            
            if len(user_text) < 2:
                # Silence / short noise detected
                self.send_json_response(200, {
                    "transcription": "",
                    "error": "blank_audio"
                })
                return
                
            self.send_json_response(200, {
                "transcription": user_text
            })
            
        except Exception as e:
            print(f"[ERROR] Error in audio transcription: {e}")
            self.send_json_response(500, {"error": f"Failed to transcribe audio: {e}"})

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, InterviewRequestHandler)
    print(f"=======================================================")
    print(f"Server running offline at:")
    print(f"http://localhost:{port}")
    print(f"=======================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server...")
        httpd.server_close()

if __name__ == "__main__":
    run()
