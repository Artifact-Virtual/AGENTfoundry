# Agent Foundry

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green)](https://docs.pytest.org/)
[![Aider](https://img.shields.io/badge/AI-Aider--Chat-brightgreen.svg)](https://github.com/paul-gauthier/aider)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black.svg)](https://ollama.ai/)

An AI-powered code generation tool that converts natural language prompts into working codebases using multi-agent workflows and iterative refinement. In other words, this is a prompt-to-product, fully hands-off automation, from prototype working code. It is designed as a compliment to the newly introduced in-house production pipeline to help automate the iteration process to develop robust systems.

## Features

- Multi-agent optimization system (MAOS) for prompt engineering
- Integration with Aider for AI-assisted code generation
- Complete project structure generation with documentation
- Support for multiple programming languages and frameworks
- Automated testing and deployment preparation

## Installation

```bash
git clone https://github.com/artifact-virtual/agent-foundry.git
cd agent-foundry
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Prerequisites

- Python 3.10+
- aider-chat
- ollama (or compatible LLM service)

## Quick Start

```bash
python cli.py --user demo --input "Create a REST API for task management"
```

## Configuration

Set up your environment:

```bash
export OLLAMA_HOST="localhost:11434"  # If using Ollama locally
```

## Project Structure

Generated projects include:
- Source code with proper structure
- Test files and configuration
- Documentation and setup instructions
- Deployment scripts when applicable

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](./LICENSE) for details.

