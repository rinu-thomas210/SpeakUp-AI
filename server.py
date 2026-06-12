from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import json
import urllib.parse
import uuid
import ollama
from faster_whisper import WhisperModel

# Global Session State
session_state = {
    "messages": [],
    "model_name": "llama3.2:1b",
    "job_role": "Python Backend Developer",
    # Built-in Mock Mode state
    "is_mock_mode": False,
    "mock_questions": [],
    "mock_current_index": 0
}

# Built-in Mock Questions and Keywords Database
MOCK_QUESTIONS = {
    "Python Backend Developer": [
        {
            "question": "Could you introduce yourself and tell me what motivates you to work specifically in Python backend development?",
            "keywords": ["experience", "motivation", "fastapi", "django", "backend", "frameworks", "python", "developer", "scalable"],
            "topic": "Introduction & Motivation"
        },
        {
            "question": "Could you explain the differences between lists and tuples in Python, and in what scenarios you would choose one over the other for memory optimization?",
            "keywords": ["mutable", "immutable", "performance", "memory", "modify", "change", "parenthesis", "bracket", "constant", "hashable", "dict key"],
            "topic": "List vs Tuple"
        },
        {
            "question": "Imagine you are tasked with designing a RESTful API that needs to serve millions of requests daily. How would you handle state, scaling, and database bottleneck optimization in Python?",
            "keywords": ["rest", "endpoint", "version", "alembic", "django", "sql", "migration", "database", "scale", "stateless", "http", "caching", "redis"],
            "topic": "RESTful API Design & Scaling"
        },
        {
            "question": "Can you share a time when you had a major technical disagreement with a team member about a backend architectural decision, and how did you resolve it?",
            "keywords": ["disagreement", "architectural", "decision", "resolve", "communication", "metrics", "test", "compromise", "team"],
            "topic": "Technical Disagreement & Collaboration"
        },
        {
            "question": "How does Python's Global Interpreter Lock (GIL) affect multi-threaded CPU-bound programs, and how would you bypass it for parallel processing?",
            "keywords": ["gil", "lock", "thread", "concurrency", "multiprocessing", "cpu", "io", "asyncio", "process", "parallel"],
            "topic": "Global Interpreter Lock (GIL)"
        },
        {
            "question": "Could you tell me about the most challenging Python backend bug or performance leak you encountered in a past project, and how did you diagnose and fix it?",
            "keywords": ["challenge", "bug", "leak", "diagnose", "fix", "memory", "profiling", "logging", "resolved", "database"],
            "topic": "Backend Debugging & Profiling"
        },
        {
            "question": "When designing middleware in frameworks like FastAPI or Django, what are the key security and lifecycle aspects you ensure are handled at the request level?",
            "keywords": ["middleware", "request", "response", "fastapi", "django", "intercept", "auth", "logging", "cors", "filter", "security"],
            "topic": "Middleware & Request Lifecycle"
        },
        {
            "question": "Suppose you have a database migration in production with millions of active rows, and you need to add a non-nullable column. How do you execute this without downtime?",
            "keywords": ["migration", "downtime", "production", "nullable", "column", "default value", "step", "phased", "alembic", "database"],
            "topic": "Zero-Downtime Database Migrations"
        },
        {
            "question": "In a fast-paced environment, how do you balance writing clean, testable code with meeting tight product deadlines?",
            "keywords": ["clean", "testable", "code", "deadline", "balance", "technical debt", "refactoring", "priority", "agile"],
            "topic": "Code Quality vs Deadlines"
        },
        {
            "question": "Finally, when integrating background task workers like Celery or Redis Queue, how do you handle tasks that fail in production to prevent data loss or silent errors?",
            "keywords": ["celery", "queue", "fail", "retry", "dead letter", "monitor", "silent error", "logging", "worker", "exception"],
            "topic": "Background Tasks & Failures"
        }
    ],
    "Frontend React Developer": [
        {
            "question": "What makes you passionate about frontend engineering, and why React in particular?",
            "keywords": ["passionate", "frontend", "react", "component", "ecosystem", "declarative", "ui", "ux", "developer"],
            "topic": "Motivation & Introduction"
        },
        {
            "question": "How does React's Virtual DOM work under the hood, and how does reconciliation optimize rendering performance?",
            "keywords": ["virtual dom", "reconciliation", "diff", "render", "update", "batch", "real dom", "ui", "performance", "patch"],
            "topic": "Virtual DOM"
        },
        {
            "question": "Imagine you are building a large dashboard with numerous real-time widgets. What state management approach (Context API, Redux, Zustand, etc.) would you select and why?",
            "keywords": ["context", "redux", "zustand", "state", "global", "prop drilling", "store", "action", "reducer", "re-render"],
            "topic": "State Management"
        },
        {
            "question": "Tell me about a time in a past project when you had to significantly optimize a React app's initial load time. What specific strategies did you implement?",
            "keywords": ["optimize", "load time", "performance", "code split", "lazy", "suspense", "bundle", "webpack", "vite", "loading"],
            "topic": "Frontend Optimization"
        },
        {
            "question": "Can you explain the differences between useMemo and useCallback, and describe a scenario where misusing them might actually degrade performance?",
            "keywords": ["usememo", "usecallback", "memoize", "function", "value", "overhead", "re-render", "dependency", "reference"],
            "topic": "useMemo & useCallback Hooks"
        },
        {
            "question": "Describe a time when a designer handed you a complex, non-standard UI flow. How did you collaborate to implement it while keeping accessibility and usability in mind?",
            "keywords": ["collaboration", "design", "accessibility", "usability", "a11y", "aria", "implementation", "ux", "designer"],
            "topic": "Cross-functional Collaboration"
        },
        {
            "question": "How do you handle error boundaries in React, and what is your strategy for logging runtime client-side exceptions in production?",
            "keywords": ["error boundary", "logging", "exception", "sentry", "catch", "crash", "production", "client-side"],
            "topic": "Error Handling & Observability"
        },
        {
            "question": "What is your approach to component styling (Vanilla CSS, CSS Modules, Styled Components, Tailwind CSS) and how do you ensure scaling design consistency?",
            "keywords": ["style", "css", "tailwind", "modules", "styled components", "consistency", "design system", "tokens"],
            "topic": "Component Styling & Design System"
        },
        {
            "question": "Suppose you have a parent component that re-renders frequently, causing its expensive child components to lag. How would you diagnose and resolve this?",
            "keywords": ["render", "frequently", "lag", "diagnose", "resolve", "react.memo", "profiler", "usememo", "state localization"],
            "topic": "Performance Debugging"
        },
        {
            "question": "Lastly, how do you approach writing clean, maintainable unit and integration tests for React components, and what tools do you prefer?",
            "keywords": ["test", "unit", "integration", "jest", "react testing library", "rtl", "mock", "assertion", "coverage"],
            "topic": "Testing Strategy"
        }
    ],
    "Full Stack Software Engineer": [
        {
            "question": "What drives your interest in full-stack development, and how do you balance frontend and backend priorities?",
            "keywords": ["interest", "full-stack", "balance", "frontend", "backend", "priorities", "skills", "features"],
            "topic": "Motivation & Introduction"
        },
        {
            "question": "Can you walk me through a full-stack architecture you've designed from scratch, explaining how the client, backend, database, and caching layers communicate?",
            "keywords": ["architecture", "frontend", "backend", "database", "cache", "redis", "nginx", "load balancer", "api", "client", "server"],
            "topic": "Full-Stack Architecture"
        },
        {
            "question": "Security is critical. How do you protect a web application from major vulnerabilities like Cross-Site Scripting (XSS), CSRF, and SQL Injection?",
            "keywords": ["csrf", "xss", "sql injection", "sanitize", "token", "validate", "cors", "https", "orm", "security"],
            "topic": "Web Application Security"
        },
        {
            "question": "Tell me about a time in a past project when a service crashed unexpectedly in production. How did you debug the issue across the full stack?",
            "keywords": ["crash", "production", "debug", "logs", "monitoring", "root cause", "database", "api", "resolved"],
            "topic": "Production Incident Debugging"
        },
        {
            "question": "What are the key factors you consider when choosing between a relational database like PostgreSQL and a NoSQL database like MongoDB for a new feature?",
            "keywords": ["relational", "nosql", "postgres", "mongodb", "schema", "acid", "scale", "join", "flexibility", "document"],
            "topic": "SQL vs NoSQL Databases"
        },
        {
            "question": "How do you implement secure, scalable authentication and authorization (JWT, sessions, OAuth) across a distributed full-stack architecture?",
            "keywords": ["auth", "jwt", "session", "oauth", "token", "cookie", "login", "header", "verify", "role"],
            "topic": "Authentication & Authorization"
        },
        {
            "question": "Describe a situation where a frontend developer and a backend developer disagreed on an API contract. How did you intervene to resolve it?",
            "keywords": ["disagreement", "api contract", "resolve", "collaboration", "swagger", "openapi", "mocking", "communication"],
            "topic": "Team Collaboration & APIs"
        },
        {
            "question": "What is your approach to CI/CD pipelines and deployment containerization? How do you ensure seamless releases?",
            "keywords": ["ci/cd", "pipeline", "docker", "github actions", "deploy", "test", "build", "jenkins", "kubernetes", "cloud"],
            "topic": "CI/CD & Containers"
        },
        {
            "question": "Imagine your system experiences a sudden spike in traffic. Where do you look first for bottlenecks, and how do you scale the frontend and backend?",
            "keywords": ["spike", "traffic", "bottleneck", "scale", "load balancer", "caching", "redis", "database", "profiling"],
            "topic": "System Scaling"
        },
        {
            "question": "Lastly, how do you maintain code quality and testing standards across both the frontend and backend codebases in a fast-paced environment?",
            "keywords": ["code quality", "testing", "standards", "linting", "unit tests", "integration", "code review", "fast-paced"],
            "topic": "Quality & Testing Standards"
        }
    ],
    "Data Scientist / AI Engineer": [
        {
            "question": "What drew you to this field, and what kind of machine learning problems excite you most?",
            "keywords": ["drew", "field", "machine learning", "problems", "excite", "algorithms", "data", "ai"],
            "topic": "Motivation & Introduction"
        },
        {
            "question": "How do you evaluate bias-variance tradeoff when a model is overfitting, and what regularization techniques do you apply to resolve it?",
            "keywords": ["bias", "variance", "tradeoff", "overfitting", "underfitting", "generalize", "complexity", "noise", "error"],
            "topic": "Bias-Variance Tradeoff"
        },
        {
            "question": "Imagine you are training a classification model on an extremely imbalanced dataset (e.g. fraud detection). How do you handle this, and what evaluation metrics do you prioritize?",
            "keywords": ["imbalanced", "oversampling", "undersampling", "smote", "class weight", "precision", "recall", "f1-score", "resample"],
            "topic": "Imbalanced Datasets"
        },
        {
            "question": "Tell me about the most challenging dataset you've worked with. How did you handle data cleaning, missing values, and feature engineering?",
            "keywords": ["challenge", "dataset", "cleaning", "missing", "feature engineering", "outliers", "transformation", "imputation"],
            "topic": "Data Wrangling & Feature Engineering"
        },
        {
            "question": "What is the difference between fine-tuning a pre-trained LLM and implementing Retrieval-Augmented Generation (RAG)? In what scenario would you choose each?",
            "keywords": ["fine-tune", "rag", "retrieval", "llm", "prompt", "vector db", "knowledge", "embedding", "context", "weights"],
            "topic": "LLM Fine-tuning vs RAG"
        },
        {
            "question": "How do you bridge the gap between building a model in a notebook and deploying it as a scalable production API?",
            "keywords": ["bridge", "notebook", "deploy", "production", "api", "docker", "fastapi", "flask", "latency", "scalability"],
            "topic": "Model Deployment & MLOps"
        },
        {
            "question": "Describe a time when a model performed well during training and validation but failed when deployed to production. What went wrong and how did you resolve it?",
            "keywords": ["validation", "failed", "production", "drift", "leakage", "data pipeline", "diagnose", "resolve"],
            "topic": "Production Model Troubleshooting"
        },
        {
            "question": "How do you explain complex machine learning model decisions to non-technical stakeholders or business leaders?",
            "keywords": ["explain", "complex", "stakeholder", "business", "shap", "lime", "interpretability", "metrics", "visualization"],
            "topic": "Model Interpretability & Communication"
        },
        {
            "question": "What is your process for tracking machine learning experiments, and how do you handle version control for datasets and models?",
            "keywords": ["experiment", "tracking", "version control", "mlflow", "dvc", "dataset", "model registry", "git"],
            "topic": "Experiment Tracking & Version Control"
        },
        {
            "question": "Lastly, how do you keep up with the rapid advancements in AI and Large Language Models, and how do you decide when to adopt a new tool or research paper?",
            "keywords": ["keep up", "advancement", "ai", "llm", "adopt", "research", "paper", "tools", "evaluation", "benchmark"],
            "topic": "Continuous Learning in AI"
        }
    ],
    "DevOps & Cloud Engineer": [
        {
            "question": "What does 'DevOps culture' mean to you, and why did you choose this field?",
            "keywords": ["devops", "culture", "collaboration", "automation", "choose", "field", "mindset"],
            "topic": "Motivation & Introduction"
        },
        {
            "question": "What is Infrastructure as Code (IaC), and how do you structure your Terraform files and state management for multi-environment cloud deployments?",
            "keywords": ["iac", "terraform", "state file", "declarative", "provider", "cloud", "resource", "yaml", "ansible"],
            "topic": "Infrastructure as Code (IaC)"
        },
        {
            "question": "Imagine you need to scale a microservices application. What are the key differences between horizontal and vertical scaling, and how do you configure autoscaling?",
            "keywords": ["horizontal", "vertical", "scaling", "autoscaling", "instance", "cpu", "load balancer", "ram", "replica"],
            "topic": "Horizontal vs Vertical Scaling"
        },
        {
            "question": "Tell me about a time when a deployment pipeline broke or a cloud service went down under your watch. How did you react, diagnose, and resolve it?",
            "keywords": ["crashed", "deployment", "pipeline", "down", "react", "diagnose", "resolve", "rollback", "logging"],
            "topic": "Incident Response"
        },
        {
            "question": "How do you secure containers in a Kubernetes cluster, and what are the best practices for managing cloud credentials and application secrets?",
            "keywords": ["container", "kubernetes", "secret", "vault", "security", "rbac", "network policy", "namespace", "image scanning"],
            "topic": "Kubernetes & Container Security"
        },
        {
            "question": "What is your strategy for setting up zero-downtime deployment strategies like Blue-Green or Canary releases in production?",
            "keywords": ["zero-downtime", "blue-green", "canary", "deployment", "traffic", "rollout", "rollback", "load balancer", "switch"],
            "topic": "Deployment Strategies"
        },
        {
            "question": "How do you set up monitoring, logging, and alerting for a distributed system using tools like Prometheus, Grafana, or the ELK stack?",
            "keywords": ["monitor", "alert", "prometheus", "grafana", "elk", "log", "metrics", "tracing", "dashboard", "alertmanager"],
            "topic": "Monitoring & Observability"
        },
        {
            "question": "Describe a time when you had to work with software developers who were resistant to adopting DevOps practices or testing pipelines. How did you win them over?",
            "keywords": ["developers", "resistant", "adopt", "win over", "collaboration", "empathy", "demonstrate value"],
            "topic": "Cultural Alignment & Collaboration"
        },
        {
            "question": "How do you optimize cloud computing costs while ensuring application performance and reliability are not compromised?",
            "keywords": ["optimize", "cost", "cloud", "billing", "performance", "reliability", "instance sizing", "spot instances"],
            "topic": "Cost Optimization"
        },
        {
            "question": "Lastly, how do you design a robust disaster recovery plan for a cloud-hosted application to guarantee high availability?",
            "keywords": ["disaster recovery", "plan", "cloud", "availability", "backup", "failover", "rto", "rpo", "replication"],
            "topic": "Disaster Recovery"
        }
    ],
    "Product Manager": [
        {
            "question": "What drives you to build products, and what makes a product 'great' in your eyes?",
            "keywords": ["drives", "build", "products", "great", "vision", "strategy", "value", "user"],
            "topic": "Motivation & Introduction"
        },
        {
            "question": "How do you prioritize a product backlog when you have competing requests from engineering, sales, customer support, and executive leadership?",
            "keywords": ["prioritize", "backlog", "rice", "moscow", "roi", "value", "effort", "customer", "stakeholder", "strategy"],
            "topic": "Backlog Prioritization"
        },
        {
            "question": "Imagine a high-priority feature you launched failed to meet its target adoption metrics. How do you analyze the failure and decide on next steps?",
            "keywords": ["failed", "adoption", "metrics", "analyze", "failure", "feedback", "post-mortem", "next steps", "iteration"],
            "topic": "Product Launch Post-mortem"
        },
        {
            "question": "Tell me about a time when you had to say 'no' to a major stakeholder or customer request. How did you manage that communication?",
            "keywords": ["say no", "stakeholder", "customer", "request", "manage", "communication", "data-backed", "priority"],
            "topic": "Stakeholder Management"
        },
        {
            "question": "How do you translate user research insights and customer pain points into clear, actionable Product Requirement Documents (PRDs) for engineers?",
            "keywords": ["user research", "interview", "pain point", "prd", "requirement", "persona", "feedback", "survey", "problem space"],
            "topic": "User Research & PRD"
        },
        {
            "question": "How do you handle situations where the engineering team estimates a feature will take three times longer than you planned? How do you negotiate scope?",
            "keywords": ["engineering", "estimates", "negotiate", "scope", "time", "milestones", "mvp", "tradeoffs"],
            "topic": "Scope & Engineering Negotiation"
        },
        {
            "question": "What metrics and key performance indicators (KPIs) do you track daily to measure the health and success of your product?",
            "keywords": ["kpi", "metrics", "feature", "retention", "engagement", "adoption", "conversion", "analytics", "okr"],
            "topic": "Product Success Metrics"
        },
        {
            "question": "Describe a time when you had to make a critical product decision with incomplete data. What was your process and outcome?",
            "keywords": ["critical", "decision", "incomplete data", "process", "outcome", "intuition", "risk mitigation", "hypothesis"],
            "topic": "Data-Incomplete Decision Making"
        },
        {
            "question": "How do you align cross-functional teams (engineering, design, marketing, sales) behind a shared product vision and roadmap?",
            "keywords": ["align", "cross-functional", "roadmap", "vision", "collaboration", "shared goals", "communication"],
            "topic": "Cross-Functional Alignment"
        },
        {
            "question": "Lastly, how do you identify new market opportunities or user needs, and how do you validate them before engineering begins development?",
            "keywords": ["market", "opportunities", "validate", "mvp", "user testing", "prototyping", "research"],
            "topic": "Opportunity Validation"
        }
    ],
    "Custom Role": [
        {
            "question": "Could you introduce yourself and highlight your core areas of expertise?",
            "keywords": ["experience", "technology", "skills", "project", "framework", "languages", "architecture", "senior", "developer", "engineer"],
            "topic": "Professional Summary"
        },
        {
            "question": "What is the most challenging technical project you've worked on recently in your capacity as a {JOB_ROLE}, and what was your role in its success?",
            "keywords": ["challenge", "project", "complex", "obstacle", "problem", "solve", "debug", "architecture", "scaling", "resolved"],
            "topic": "Technical Challenge"
        },
        {
            "question": "How do you keep up with the latest trends, frameworks, and best practices in your field?",
            "keywords": ["learn", "blog", "newsletter", "documentation", "github", "course", "trend", "best practices", "community", "podcast"],
            "topic": "Continuous Learning"
        },
        {
            "question": "Tell me about a time when you had to adapt to a sudden change in project requirements or priorities. How did you handle it?",
            "keywords": ["adapt", "sudden change", "requirements", "priorities", "agile", "flexible", "communication", "re-prioritization"],
            "topic": "Adaptability"
        },
        {
            "question": "How do you approach collaboration with other cross-functional roles, such as developers, designers, or product owners, to deliver high-quality work?",
            "keywords": ["collaboration", "communication", "team", "agile", "scrum", "git", "review", "designer", "pm", "feedback"],
            "topic": "Team Collaboration"
        },
        {
            "question": "Can you describe your ideal development workflow, including the tools you use for coding, testing, and deployment?",
            "keywords": ["workflow", "tools", "git", "docker", "ci/cd", "test", "deploy", "editor", "ide", "methodology"],
            "topic": "Development Workflow"
        },
        {
            "question": "Describe a time you made a technical mistake in your work. What did you learn from it and how did you prevent it from happening again?",
            "keywords": ["mistake", "error", "lesson", "prevent", "post-mortem", "technical debt", "code review"],
            "topic": "Lessons Learned"
        },
        {
            "question": "How do you approach solving a complex technical problem that you haven't encountered before? What is your step-by-step process?",
            "keywords": ["solve", "complex problem", "step-by-step", "research", "debugging", "isolation", "first principles"],
            "topic": "Problem Solving"
        },
        {
            "question": "How do you ensure the quality, security, and performance of the deliverables you produce in your daily role?",
            "keywords": ["quality", "security", "performance", "deliverables", "best practices", "testing", "benchmarking"],
            "topic": "Quality Assurance"
        },
        {
            "question": "Lastly, what do you hope to achieve in this mock interview today, and what areas of your skills as a {JOB_ROLE} are you most looking to test?",
            "keywords": ["achieve", "mock interview", "test", "skills", "practice", "evaluation", "confidence"],
            "topic": "Interview Expectations"
        }
    ]
}

def evaluate_mock_answer(user_answer):
    idx = session_state["mock_current_index"]
    questions = session_state["mock_questions"]
    
    if idx >= len(questions):
        return "Interview is already complete."
        
    q_data = questions[idx]
    keywords = q_data.get("keywords", [])
    
    # Normalize answer
    clean_answer = user_answer.lower()
    words = clean_answer.split()
    word_count = len(words)
    
    matched = []
    for kw in keywords:
        if kw in clean_answer:
            matched.append(kw)
            
    match_count = len(matched)
    
    if word_count < 8:
        feedback = "Your response is extremely brief. In a realistic interview scenario, it is critical to elaborate, provide technical details, or share past project experiences to fully demonstrate your expertise."
    elif word_count < 22:
        feedback = "Your explanation is a bit brief. While you touched on the basic concept, it would be much stronger if you expanded on the implementation details and highlighted how you apply these patterns in practice."
    else:
        if match_count > 0:
            feedback = f"That is a solid and well-structured answer. You did a great job explaining the concept and correctly highlighted key industry terms like {', '.join(matched[:4])}, which demonstrates your direct practical knowledge."
        else:
            feedback = f"Your answer has a good flow, though you could make it much more impactful by referencing specific terms and concepts such as {', '.join(keywords[:3])} to ground your explanation in standard terminology."
            
    return feedback

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
        elif path == "/api/answer-audio":
            self.handle_answer_audio()
        else:
            self.send_error(404, "Endpoint Not Found")

    def read_json_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))

    def handle_get_models(self):
        models = ["Mock Interviewer (No Ollama required)"]
        try:
            models_response = ollama.list()
            ollama_models = []
            if isinstance(models_response, dict) and 'models' in models_response:
                for m in models_response['models']:
                    if hasattr(m, 'model'):
                        ollama_models.append(m.model)
                    elif isinstance(m, dict) and 'model' in m:
                        ollama_models.append(m['model'])
                    elif isinstance(m, dict) and 'name' in m:
                        ollama_models.append(m['name'])
            elif hasattr(models_response, 'models'):
                for m in models_response.models:
                    if hasattr(m, 'model'):
                        ollama_models.append(m.model)
                    elif hasattr(m, 'name'):
                        ollama_models.append(m.name)
            
            # Deduplicate and extend
            ollama_models = list(set(ollama_models))
            models.extend(ollama_models)
            self.send_json_response(200, {"models": models})
        except Exception as e:
            print(f"[WARN] Failed to list Ollama models: {e}")
            self.send_json_response(200, {"models": models, "error": str(e)})

    def handle_start_interview(self):
        try:
            data = self.read_json_body()
            job_role = data.get("job_role", "Python Backend Developer").strip()
            model_name = data.get("model_name", "llama3.2:1b").strip()
            
            session_state["job_role"] = job_role
            session_state["model_name"] = model_name
            
            # Reset Mock Mode State
            session_state["is_mock_mode"] = False
            session_state["mock_questions"] = []
            session_state["mock_current_index"] = 0
            
            system_prompt = f"""
You are an expert technical interviewer conducting a realistic, dynamic mock interview for a {job_role} position. Your goal is to evaluate the candidate holistically by naturally interweaving technical, behavioral, project-based, and situational questions throughout the conversation.

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
            session_state["messages"] = [{"role": "system", "content": system_prompt}]
            
            print(f"[START] Starting mock interview for: {job_role} (Model: {model_name})")
            
            # Use Built-in Mock Mode directly if selected
            if model_name == "Mock Interviewer (No Ollama required)":
                session_state["is_mock_mode"] = True
                role_key = job_role if job_role in MOCK_QUESTIONS else "Custom Role"
                session_state["mock_questions"] = MOCK_QUESTIONS[role_key]
                first_q = session_state["mock_questions"][0]["question"]
                
                ai_response = f"Hello! Welcome to your mock interview for the {job_role} position. Let's get started. {first_q}"
                session_state["messages"].append({"role": "assistant", "content": ai_response})
                
                self.send_json_response(200, {
                    "question": ai_response,
                    "status": "started",
                    "fallback": False
                })
                return
                
            # Otherwise, try to use local Ollama
            try:
                response = ollama.chat(model=model_name, messages=session_state["messages"])
                ai_response = response['message']['content']
                session_state["messages"].append({"role": "assistant", "content": ai_response})
                
                self.send_json_response(200, {
                    "question": ai_response,
                    "status": "started",
                    "fallback": False
                })
            except Exception as ollama_err:
                print(f"[WARN] Failed to connect to Ollama: {ollama_err}. Falling back to Built-in Mock Interviewer.")
                session_state["is_mock_mode"] = True
                role_key = job_role if job_role in MOCK_QUESTIONS else "Custom Role"
                session_state["mock_questions"] = MOCK_QUESTIONS[role_key]
                first_q = session_state["mock_questions"][0]["question"]
                
                ai_response = f"Hello! Welcome to your mock interview for the {job_role} position. Let's get started. {first_q}"
                session_state["messages"].append({"role": "assistant", "content": ai_response})
                
                self.send_json_response(200, {
                    "question": ai_response,
                    "status": "started",
                    "fallback": True,
                    "fallback_reason": f"Failed to connect to local Ollama ({ollama_err}). Running in offline Built-in Mock Interviewer mode."
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
                
            if session_state.get("is_mock_mode", False):
                feedback = evaluate_mock_answer(user_answer)
                questions = session_state["mock_questions"]
                
                # Advance to next question using modulo for infinite cycling
                session_state["mock_current_index"] = (session_state["mock_current_index"] + 1) % len(questions)
                next_q = questions[session_state["mock_current_index"]]["question"]
                ai_response = f"{feedback}\n\nNext question:\n{next_q}"
                    
                session_state["messages"].append({"role": "user", "content": user_answer})
                session_state["messages"].append({"role": "assistant", "content": ai_response})
                
                self.send_json_response(200, {
                    "transcription": user_answer,
                    "response": ai_response
                })
                return
                
            session_state["messages"].append({"role": "user", "content": user_answer})
            print(f"[INFO] Processed user typed answer. Sending to Ollama...")
            
            response = ollama.chat(model=session_state["model_name"], messages=session_state["messages"])
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

    def handle_answer_audio(self):
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
                    "response": "I couldn't hear you clearly. Could you please re-record or type your answer?",
                    "error": "blank_audio"
                })
                return
                
            # Feed transcription to Ollama
            if session_state.get("is_mock_mode", False):
                feedback = evaluate_mock_answer(user_text)
                questions = session_state["mock_questions"]
                
                # Advance to next question using modulo for infinite cycling
                session_state["mock_current_index"] = (session_state["mock_current_index"] + 1) % len(questions)
                next_q = questions[session_state["mock_current_index"]]["question"]
                ai_response = f"{feedback}\n\nNext question:\n{next_q}"
                    
                session_state["messages"].append({"role": "user", "content": user_text})
                session_state["messages"].append({"role": "assistant", "content": ai_response})
                
                self.send_json_response(200, {
                    "transcription": user_text,
                    "response": ai_response
                })
                return
                
            session_state["messages"].append({"role": "user", "content": user_text})
            print(f"[INFO] Sending transcription to Ollama...")
            
            response = ollama.chat(model=session_state["model_name"], messages=session_state["messages"])
            ai_response = response['message']['content']
            print(f"[OK] Ollama responded. Sending reply to browser.")
            
            session_state["messages"].append({"role": "assistant", "content": ai_response})
            
            self.send_json_response(200, {
                "transcription": user_text,
                "response": ai_response
            })
            
        except Exception as e:
            print(f"[ERROR] Error in audio answer: {e}")
            self.send_json_response(500, {"error": f"Failed to transcribe or process audio: {e}"})

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
