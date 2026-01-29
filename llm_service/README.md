# LLM Service

This directory contains the optional local LLM service used for
forecast interpretation and natural-language explanations.

## Purpose
- Generates human-readable interpretations of price forecasts and system analytics including fault diagnosis of the heating system
- Isolated from the main control system
- Can be disabled without affecting control logic

## Setup

```bash
cd llm_service
python3 -m venv .venv-llm
source .venv-llm/bin/activate
pip install -r requirements.txt
bash start.sh
```

## Running llm_service in a separate virtual environment:

create .config/systemd/user/ and copy llm_service.service to it 