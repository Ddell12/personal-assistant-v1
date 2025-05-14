# Personal Assistant V1

A modular personal AI assistant with Realtime API access, vector store integration, and various tools.

## Features

- Modular agent architecture with pluggable components
- Vector store integration for document search and RAG capabilities
- OpenAI integration for advanced AI capabilities
- Real-time speech-to-speech functionality using OpenAI's Realtime API
- WebSocket communication for real-time interactions

## Project Structure

- `personal_assistant/`: Main package
  - `agent.py`: Core agent implementation
  - `memory_service.py`: Memory management
  - `vector_stores/`: Vector store implementations
  - `tools/`: Tool functions for the agent
  - `sub_agents/`: Specialized agent implementations
  - `shared_libraries/`: Common utilities

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate the virtual environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/Mac: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Set up environment variables in `.env` file
6. Run the application: `python -m personal_assistant.main`

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

MIT
