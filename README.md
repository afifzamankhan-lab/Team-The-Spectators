# QueueStorm Investigator Copilot
**SUST CSE Carnival 2026: Codex Community Hackathon (Online Preliminary Round)**
This repository contains the backend API service for the **QueueStorm Investigator** challenge. This system operates as an internal support copilot built specifically to analyze massive influxes of customer complaints during high-traffic digital finance campaigns.
Instead of acting as a simple text classifier, this service is a true complaint investigator. It ingests raw customer text and cross-references it with recent account transaction history data to programmatically determine what is true, route the case to the correct operations team, flag high-risk situations, and draft an officially compliant response.
##  Tech Stack & Architecture
 * **Language:** Python 3.11
 * **Framework:** FastAPI (Selected for rapid development and native Pydantic data validation)
 * **Server:** Uvicorn
 * **AI Engine:** Google Gemini API (google-generativeai SDK)
##  Setup Instructions & Runbook
This service exposes two required endpoints: GET /health and POST /analyze-ticket. If our live server URL is unreachable, use either of the fallback methods below to reproduce the environment locally.
### Option A: Running via Docker (Preferred Fallback)
Our Docker setup uses a lightweight, optimized python:3.11-slim base image to maintain a small memory footprint safely under the 1GB limit.
 1. **Build the container image:**
   ```bash
   docker build -t hackathon-team .
   
   ```
 2. **Run the container:**
   *(Note: Real credentials are never baked into our images or codebase. You must pass your Gemini API key dynamically via environment variables during runtime).*
   ```bash
   docker run -p 8000:8000 -e GEMINI_API_KEY="your_actual_api_studio_key" hackathon-team
   
   ```
### Option B: Running locally with native Python
 1. Clone this repository and open your terminal in the project root folder.
 2. Install all required dependencies from the package file:
   ```bash
   pip install -r requirements.txt
   
   ```
 3. Create a local environment file named .env in the root folder using .env.example as a template:
   ```env
   GEMINI_API_KEY=AIzaSyYourKeyHere
   
   ```
 4. Start the server interface bound to 0.0.0.0 on port 8000:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000
   
   ```
##  MODELS Usage & AI Approach
 * **Model Employed:** gemini-1.5-flash
 * **Execution Environment:** Hosted externally via the Google AI Studio platform.
 * **Selection & Cost Reasoning:** The evaluation harness enforces a strict 30-second response window for POST /analyze-ticket. We explicitly chose the flash model over heavier options because of its rapid token generation speed and ultra-low latency. This allows our backend to comfortably complete complex context reasoning and format JSON well within the time budget while minimizing API quota consumption.
### Hybrid Engineering Methodology
To guarantee maximum accuracy under scale, we designed a **Hybrid Rule + AI system**:
 1. **Contextual Reasoning:** The raw complaint text (which can arrive in English, Bangla, or unstructured Banglish) and transaction history rows are structured into an investigative prompt. The LLM performs the language understanding, pattern-matching, and contextual summary generation.
 2. **Strict Enum Mapping:** The LLM is instructed to output structurally sound JSON. However, to protect against hallucinations causing schema violations, our Python backend intercepts the response and runs it through an explicit enum coercion mapper to ensure exact matches with the taxonomy.
 3. **Deterministic Safety Enforcement:** The final output parameters are subjected to rigid, rule-based regex and token sanitization layers before being sent back to the client.
## Safety Logic & Guardrails
Our system treats user inputs as untrusted data and applies robust safety filters to counter prompt injections and prevent severe score penalties:
 * **Credential Exposure Mitigation:** Our post-processing filter scans the compiled customer_reply. If keywords