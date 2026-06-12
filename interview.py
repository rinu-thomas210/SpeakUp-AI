import ollama
import pyttsx3
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import os
import sys
import time

# TTS initialization removed – will initialize per call


# Fallback pyttsx3 engine (will be re-instantiated per call)



# Engine will be created per speak call


# CHANGE THIS to your exact model name from 'ollama list'
MODEL_NAME = "YOUR_MODEL_NAME_HERE" 

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

2. CONVERSATIONAL LOOP: Ask ONLY ONE core question at a time. When the user provides an answer, you must:
   a. Briefly acknowledge or evaluate their response (1-2 sentences max).
   b. Occasionally ask a SHORT follow-up to dig deeper into something interesting or weak in their answer.
   c. Then, seamlessly transition to the NEXT interview question from a DIFFERENT category than the previous one.

3. TONE & PERSONA: Speak like a real senior interviewer—professional, observant, and engaging. Use natural transitions like "That's a great point, now switching gears..." or "Interesting. Let me challenge you on this..."

4. ADAPTIVE DIFFICULTY: If the candidate answers well, increase difficulty. If they struggle, offer a small hint and move on.

5. CONCISENESS: Keep your responses under 4 sentences total (feedback + next question combined).

6. NEVER BREAK CHARACTER. Start immediately by introducing yourself and asking the first question.
"""

# Built-in Mock Questions and Keywords Database
MOCK_QUESTIONS = {
    "Python Backend Developer": [
        {
            "question": "Could you explain the difference between a list and a tuple in Python, and when you would choose one over the other?",
            "keywords": ["mutable", "immutable", "performance", "memory", "modify", "change", "parenthesis", "bracket", "constant", "hashable", "dict key"],
            "topic": "List vs Tuple"
        },
        {
            "question": "How does Python's Global Interpreter Lock (GIL) affect multi-threading, and how would you bypass it for CPU-bound tasks?",
            "keywords": ["gil", "lock", "thread", "concurrency", "multiprocessing", "cpu", "io", "asyncio", "process", "parallel"],
            "topic": "Global Interpreter Lock (GIL)"
        },
        {
            "question": "What is your approach to designing a scalable RESTful API, and how do you handle database migrations in production?",
            "keywords": ["rest", "endpoint", "version", "alembic", "django", "sql", "migration", "database", "scale", "stateless", "http"],
            "topic": "RESTful API Design & Migrations"
        },
        {
            "question": "Can you explain how middleware works in a web framework like Django or FastAPI, and give an example of a custom middleware you've written?",
            "keywords": ["middleware", "request", "response", "fastapi", "django", "intercept", "auth", "logging", "cors", "filter"],
            "topic": "Web Framework Middleware"
        },
        {
            "question": "How do you optimize slow database queries in a Python backend application?",
            "keywords": ["index", "query", "select_related", "prefetch_related", "orm", "slow", "profiling", "explain", "cache", "redis", "optimize"],
            "topic": "Database Query Optimization"
        }
    ],
    "Frontend React Developer": [
        {
            "question": "What are the main differences between class components and functional components with hooks, and why did React move towards hooks?",
            "keywords": ["class", "functional", "hooks", "state", "lifecycle", "componentdidmount", "useeffect", "usestate", "boilerplate", "this"],
            "topic": "Class vs Functional Components"
        },
        {
            "question": "Can you explain how React's Virtual DOM works and how it optimizes rendering performance?",
            "keywords": ["virtual dom", "reconciliation", "diff", "render", "update", "batch", "real dom", "ui", "performance", "patch"],
            "topic": "Virtual DOM"
        },
        {
            "question": "How do you manage global state in a large-scale React application, and when would you choose Context API over Redux or Zustand?",
            "keywords": ["context", "redux", "zustand", "state", "global", "prop drilling", "store", "action", "reducer", "re-render"],
            "topic": "Global State Management"
        },
        {
            "question": "What is the difference between useMemo and useCallback, and can you give a scenario where using them might actually degrade performance?",
            "keywords": ["usememo", "usecall", "memoize", "function", "value", "overhead", "re-render", "dependency", "reference"],
            "topic": "useMemo & useCallback Hooks"
        },
        {
            "question": "How do you optimize the loading time and performance of a React application (e.g., code splitting, lazy loading)?",
            "keywords": ["code split", "lazy", "suspense", "bundle", "webpack", "vite", "loading", "size", "import", "performance"],
            "topic": "Frontend Performance Optimization"
        }
    ],
    "Full Stack Software Engineer": [
        {
            "question": "Can you describe a full-stack architecture you have designed, including the frontend, backend, database, and caching layers?",
            "keywords": ["architecture", "frontend", "backend", "database", "cache", "redis", "nginx", "load balancer", "api", "client", "server"],
            "topic": "Full-Stack System Architecture"
        },
        {
            "question": "How do you ensure security in your web applications, specifically against CSRF, XSS, and SQL injection?",
            "keywords": ["csrf", "xss", "sql injection", "sanitize", "token", "validate", "cors", "https", "orm", "security"],
            "topic": "Web Application Security"
        },
        {
            "question": "What are the pros and cons of using a relational database (like PostgreSQL) versus a NoSQL database (like MongoDB) in a project?",
            "keywords": ["relational", "nosql", "postgres", "mongodb", "schema", "acid", "scale", "join", "flexibility", "document"],
            "topic": "SQL vs NoSQL Databases"
        },
        {
            "question": "How do you handle authentication and authorization across the frontend and backend (e.g., JWT, sessions, OAuth)?",
            "keywords": ["auth", "jwt", "session", "oauth", "token", "cookie", "login", "header", "verify", "role"],
            "topic": "Authentication & Authorization"
        },
        {
            "question": "Can you walk me through your CI/CD pipeline setup for deploying a full-stack application?",
            "keywords": ["ci/cd", "pipeline", "docker", "github actions", "deploy", "test", "build", "jenkins", "kubernetes", "cloud"],
            "topic": "CI/CD & Deployment Pipelines"
        }
    ],
    "Data Scientist / AI Engineer": [
        {
            "question": "What is the difference between supervised and unsupervised learning, and how do you decide which algorithm to use for a new dataset?",
            "keywords": ["supervised", "unsupervised", "label", "clustering", "regression", "classification", "data", "kmeans", "model"],
            "topic": "Supervised vs Unsupervised Learning"
        },
        {
            "question": "How do you handle imbalanced datasets when training a machine learning model?",
            "keywords": ["imbalanced", "oversampling", "undersampling", "smote", "class weight", "precision", "recall", "f1-score", "resample"],
            "topic": "Imbalanced Datasets"
        },
        {
            "question": "Can you explain the bias-variance tradeoff and how it affects model generalization?",
            "keywords": ["bias", "variance", "tradeoff", "overfitting", "underfitting", "generalize", "complexity", "noise", "error"],
            "topic": "Bias-Variance Tradeoff"
        },
        {
            "question": "What is the difference between fine-tuning a pre-trained LLM and using Retrieval-Augmented Generation (RAG)?",
            "keywords": ["fine-tune", "rag", "retrieval", "llm", "prompt", "vector db", "knowledge", "embedding", "context", "weights"],
            "topic": "LLM Fine-tuning vs RAG"
        },
        {
            "question": "How do you evaluate the performance of a recommendation system or a classification model (e.g., ROC-AUC, F1-score)?",
            "keywords": ["roc-auc", "f1-score", "precision", "recall", "accuracy", "recommendation", "collaborative filtering", "metrics", "matrix"],
            "topic": "Model Evaluation Metrics"
        }
    ],
    "DevOps & Cloud Engineer": [
        {
            "question": "What is Infrastructure as Code (IaC), and how do you use tools like Terraform to manage cloud resources?",
            "keywords": ["iac", "terraform", "state file", "declarative", "provider", "cloud", "resource", "yaml", "ansible"],
            "topic": "Infrastructure as Code (IaC)"
        },
        {
            "question": "Can you explain the difference between horizontal and vertical scaling, and how you would configure autoscaling in AWS or GCP?",
            "keywords": ["horizontal", "vertical", "scaling", "autoscaling", "instance", "cpu", "load balancer", "ram", "replica"],
            "topic": "Horizontal vs Vertical Scaling"
        },
        {
            "question": "How do you secure containers in a Kubernetes cluster, and what are some best practices for managing secret values?",
            "keywords": ["container", "kubernetes", "secret", "vault", "security", "rbac", "network policy", "namespace", "image scanning"],
            "topic": "Kubernetes & Container Security"
        },
        {
            "question": "What is your strategy for setting up zero-downtime deployments (e.g., blue-green, canary)?",
            "keywords": ["zero-downtime", "blue-green", "canary", "deployment", "traffic", "rollout", "rollback", "load balancer", "switch"],
            "topic": "Deployment Strategies"
        },
        {
            "question": "How do you monitor and set up alerts for a distributed microservices architecture (e.g., Prometheus, Grafana, ELK stack)?",
            "keywords": ["monitor", "alert", "prometheus", "grafana", "elk", "log", "metrics", "tracing", "dashboard", "alertmanager"],
            "topic": "Monitoring & Observability"
        }
    ],
    "Product Manager": [
        {
            "question": "How do you prioritize a product backlog when you have competing requests from engineering, sales, and customers?",
            "keywords": ["prioritize", "backlog", "rice", "moscow", "roi", "value", "effort", "customer", "stakeholder", "strategy"],
            "topic": "Backlog Prioritization"
        },
        {
            "question": "Can you describe a time when a product launch didn't go as planned, and how you handled it?",
            "keywords": ["launch", "failure", "rollback", "metrics", "post-mortem", "communication", "users", "mitigate", "lesson"],
            "topic": "Product Launch Post-mortem"
        },
        {
            "question": "How do you define and track key performance indicators (KPIs) for a new feature launch?",
            "keywords": ["kpi", "metrics", "feature", "retention", "engagement", "adoption", "conversion", "analytics", "okr"],
            "topic": "Feature Success Metrics"
        },
        {
            "question": "How do you handle disagreements with engineering lead or design lead on product scope or user experience?",
            "keywords": ["disagreement", "engineer", "designer", "scope", "ux", "compromise", "data-driven", "user testing", "collaboration"],
            "topic": "Cross-Functional Alignment"
        },
        {
            "question": "Can you walk me through how you conduct user research and translate customer pain points into product requirements?",
            "keywords": ["user research", "interview", "pain point", "prd", "requirement", "persona", "feedback", "survey", "problem space"],
            "topic": "User Research & PRD"
        }
    ],
    "Custom Role": [
        {
            "question": "Could you start by summarizing your experience as a candidate and some of the key technologies you work with?",
            "keywords": ["experience", "technology", "skills", "project", "framework", "languages", "architecture", "senior", "developer", "engineer"],
            "topic": "Professional Summary"
        },
        {
            "question": "What is the most challenging technical project you've worked on recently, and how did you overcome the obstacles?",
            "keywords": ["challenge", "project", "complex", "obstacle", "problem", "solve", "debug", "architecture", "scaling", "resolved"],
            "topic": "Technical Challenge"
        },
        {
            "question": "How do you keep up with the latest trends, frameworks, and best practices in your area of expertise?",
            "keywords": ["learn", "blog", "newsletter", "documentation", "github", "course", "trend", "best practices", "community", "podcast"],
            "topic": "Continuous Learning"
        },
        {
            "question": "How do you approach collaboration with other roles, such as developers, designers, or product owners, to deliver high-quality work?",
            "keywords": ["collaboration", "communication", "team", "agile", "scrum", "git", "review", "designer", "pm", "feedback"],
            "topic": "Team Collaboration"
        },
        {
            "question": "Can you describe your ideal workflow and tools for developing, testing, and deploying your work?",
            "keywords": ["workflow", "tools", "git", "docker", "ci/cd", "test", "deploy", "editor", "ide", "methodology"],
            "topic": "Development Workflow"
        }
    ]
}

def start_interview():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    use_mock_mode = False
    mock_current_index = 0
    mock_questions = []
    
    print("-----------------------------------------------------------")
    print("Fetching First Question from Llama...")
    try:
        response = ollama.chat(model='llama3.2:1b', messages=messages)
        ai_response = response['message']['content']
    except Exception as e:
        print(f"\n❌ Ollama Connection Error: {e}")
        choice = input("⚠️ Would you like to run the interview using the Built-in Mock Mode? (y/n): ").strip().lower()
        if choice == 'y':
            use_mock_mode = True
            role_key = JOB_ROLE if JOB_ROLE in MOCK_QUESTIONS else "Custom Role"
            mock_questions = MOCK_QUESTIONS[role_key]
            first_q = mock_questions[0]["question"]
            ai_response = f"Hello! Welcome to your mock interview for the {JOB_ROLE} position. Let's get started. {first_q}"
        else:
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
            
            if use_mock_mode:
                q_data = mock_questions[mock_current_index]
                keywords = q_data.get("keywords", [])
                
                clean_answer = user_answer.lower()
                words = clean_answer.split()
                word_count = len(words)
                
                matched = []
                for kw in keywords:
                    if kw in clean_answer:
                        matched.append(kw)
                
                # Generate natural feedback (no scores)
                if word_count < 8:
                    feedback = "Your response is extremely brief. In a realistic interview scenario, it is critical to elaborate, provide technical details, or share past project experiences to fully demonstrate your expertise."
                elif word_count < 22:
                    feedback = "Your explanation is a bit brief. While you touched on the basic concept, it would be much stronger if you expanded on the implementation details and highlighted how you apply these patterns in practice."
                else:
                    if len(matched) > 0:
                        feedback = f"That is a solid and well-structured answer. You did a great job explaining the concept and correctly highlighted key industry terms like {', '.join(matched[:4])}, which demonstrates your direct practical knowledge."
                    else:
                        feedback = f"Your answer has a good flow, though you could make it much more impactful by referencing specific terms and concepts such as {', '.join(keywords[:3])} to ground your explanation in standard terminology."
                
                # Advance to next question using modulo for infinite cycling
                mock_current_index = (mock_current_index + 1) % len(mock_questions)
                next_q = mock_questions[mock_current_index]["question"]
                ai_response = f"{feedback}\n\nNext question:\n{next_q}"
                    
                speak(ai_response)
                messages.append({"role": "assistant", "content": ai_response})
            else:
                print("\nSending your text to Llama for evaluation...")
                response = ollama.chat(model='llama3.2:1b', messages=messages)
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