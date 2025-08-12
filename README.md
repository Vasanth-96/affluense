# Project Setup Instructions

# Prerequisites: Refer to the project documentation for complete details
# https://docs.google.com/document/d/1HWardg4c3YCU2uUfSnzN3Fc7VG-10crbCSe6vJWvE74/edit?tab=t.0

# 1. Python Environment Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Ollama Installation and Model Setup
# Install Ollama first (visit https://ollama.ai for installation instructions)

# Pull the Mistral model (corrected spelling)
ollama pull mistral:latest

# Start Ollama server
ollama serve

# 3. Start the Application (run in a new terminal)
uvicorn main:app --reload --workers 6
