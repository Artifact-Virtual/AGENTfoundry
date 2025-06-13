#!/usr/bin/env python3
"""
Agent Foundry Setup Script

Automates the installation and configuration of Agent Foundry
with all necessary dependencies and integrations.
"""

import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner"""
    banner = """
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                        AGENT FOUNDRY SETUP                                   │
    │                                                                              │
    │    Prompt-to-Product Pipeline                                               │
    │    MAOS Protocol Integration                                                │
    │    Multi-Iterative Code Generation                                          │
    │                                                                              │
    │    Setting up your development environment...                                │
    └──────────────────────────────────────────────────────────────────────────────┘
    """
    print(banner)

def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("ERROR: Python 3.10+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"OK: Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"Running {description}...")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        print(f"OK: {description} completed")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python Dependencies")
    print("=" * 50)
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Install aider-chat specifically
    if not run_command("pip install aider-chat", "Installing aider-chat"):
        print("WARNING: Aider installation failed - enhanced features may not work")
    
    return True

def setup_workspace():
    """Setup workspace directories"""
    print("\nSetting up Workspace")
    print("=" * 50)
    
    directories = [
        "workspace",
        "workspace/exports",
        "workspace/templates",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"OK: Created directory: {directory}")
    
    return True

def check_external_tools():
    """Check for external tools"""
    print("\nChecking External Tools")
    print("=" * 50)
    
    tools = {
        "git": "git --version",
        "node": "node --version",
        "npm": "npm --version"
    }
    
    for tool, command in tools.items():
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            version = result.stdout.strip()
            print(f"OK: {tool}: {version}")
        except Exception:
            print(f"WARNING: {tool}: Not found (optional)")

def setup_environment():
    """Setup environment configuration"""
    print("\nEnvironment Configuration")
    print("=" * 50)
    
    env_template = """
# Agent Foundry Environment Configuration
# Copy this to .env and update with your API keys

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic Configuration  
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Aider Configuration
AIDER_MODEL=gpt-4

# Workspace Configuration
WORKSPACE_DIR=./workspace
MAX_ITERATIONS=10

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/agent_foundry.log
"""
    
    env_file = Path(".env.template")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_template)
        print("OK: Created .env.template")
        print("NOTE: Copy .env.template to .env and add your API keys")
    else:
        print("OK: .env.template already exists")

def create_example_configs():
    """Create example configuration files"""
    print("\nCreating Example Configurations")
    print("=" * 50)
    
    # Example project specification
    example_project = {
        "name": "example_api",
        "type": "api",
        "description": "REST API for task management",
        "features": [
            "User authentication",
            "CRUD operations",
            "Database integration",
            "API documentation",
            "Test coverage"
        ],
        "tech_stack": {
            "backend": "Node.js + Express",
            "database": "PostgreSQL",
            "testing": "Jest",
            "docs": "Swagger"
        }
    }
    
    import json
    
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    with open(examples_dir / "project_spec.json", "w") as f:
        json.dump(example_project, f, indent=2)
    
    print("OK: Created examples/project_spec.json")

def run_quick_test():
    """Run a quick test to verify setup"""
    print("\nRunning Quick Test")
    print("=" * 50)
    
    test_script = """
import sys
sys.path.append('./core')

try:
    from core.maos_protocol import MAOSProtocol
    from core.aider_integration import AiderConfig
    print("OK: Core modules imported successfully")
    
    # Test MAOS Protocol
    maos = MAOSProtocol()
    print("OK: MAOS Protocol initialized")
    
    # Test Aider Config
    config = AiderConfig()
    print("OK: Aider Config initialized")
    
    print("Setup verification completed successfully!")
    
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    
except Exception as e:
    print(f"ERROR: {e}")
"""
    
    test_file = Path("test_setup.py")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_script)
    
    if run_command("python test_setup.py", "Running setup verification"):
        test_file.unlink()  # Delete test file
        return True
    else:
        print("WARNING: Setup verification failed - check dependencies")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\nUsage Instructions")
    print("=" * 50)
    
    instructions = """
Agent Foundry is now set up! Here's how to use it:

Basic Usage:
   python cli.py --user demo --input "Create a REST API"

Enhanced Mode (with MAOS + Aider):
   python cli.py --user demo --input "Create a web app" --enhanced

Complex Projects:
   python cli.py --user demo --input "Build microservices" --enhanced --complex

Run Demo:
   python demo.py

Configuration:
   1. Copy .env.template to .env
   2. Add your OpenAI/Anthropic API keys
   3. Configure workspace settings

Documentation:
   - README.md: Complete feature overview
   - examples/: Example configurations
   - core/: Source code documentation

Support:
   - Check logs/ directory for detailed logs
   - Review examples/ for configuration templates
   - See README.md for troubleshooting
"""
    
    print(instructions)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("ERROR: Dependency installation failed")
        sys.exit(1)
    
    # Setup workspace
    setup_workspace()
    
    # Check external tools
    check_external_tools()
    
    # Setup environment
    setup_environment()
    
    # Create examples
    create_example_configs()
    
    # Run test
    test_passed = run_quick_test()
    
    # Print instructions
    print_usage_instructions()
    
    print("\n" + "=" * 80)
    if test_passed:
        print("SUCCESS: Agent Foundry setup completed successfully!")
    else:
        print("WARNING: Setup completed with warnings - check configuration")
    print("=" * 80)

if __name__ == "__main__":
    main()
