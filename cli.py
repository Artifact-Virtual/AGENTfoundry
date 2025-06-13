import argparse
import asyncio
from core.agent import EnhancedAgent, Agent

def main():
    parser = argparse.ArgumentParser(description="Agent Foundry - Surgical Prompt-to-Product Pipeline")
    parser.add_argument("--user", required=True, help="User ID for workspace management")
    parser.add_argument("--input", required=True, help="Input prompt or requirement")
    parser.add_argument("--model", default="gpt-4o", help="LLM model to use (default: gpt-4)")
    parser.add_argument("--api-key", help="API key for the LLM service")
    parser.add_argument("--enhanced", action="store_true", help="Use enhanced agent with MAOS and Aider")
    parser.add_argument("--complex", action="store_true", help="Create complex project with multi-modal orchestration")
    parser.add_argument("--package-format", default="zip", choices=["zip", "tar.gz"], help="Package format for output")
    
    args = parser.parse_args()

    if args.enhanced:
        # Use enhanced agent with full capabilities
        agent = EnhancedAgent(args.user, model=args.model, api_key=args.api_key)
        print("Using Enhanced Agent with MAOS Protocol and Aider Integration")
        
        if args.complex:
            # Complex project creation
            project_spec = {
                "name": f"{args.user}_complex_project",
                "description": args.input,
                "type": "complex"
            }
            result = asyncio.run(agent.create_complex_project(project_spec))
            print(f"\n[COMPLEX PROJECT RESULT]\n{result}\n")
        else:
            # Standard enhanced workflow
            output = asyncio.run(agent.run_phase(args.input))
            print(f"\n[{agent.session['current_phase'].upper()} RESULT]\n{output}\n")

            if agent.session['current_phase'] == "mvp":
                build_result = asyncio.run(agent.build_project())
                if build_result["success"]:
                    print("Complete project built and packaged!")
                    print(f"Project: {build_result['project_path']}")
                    print(f"Package: {build_result['package_path']}")
                else:
                    print(f"Project build failed: {build_result.get('error', 'Unknown error')}")
    else:
        # Use legacy agent for backward compatibility
        agent = Agent(args.user)
        print("Using Legacy Agent (basic functionality)")
        
        output = agent.run_phase(args.input)
        print(f"\n[{agent.session['current_phase'].upper()} RESULT]\n{output}\n")

        if agent.session['current_phase'] == "mvp":
            build_result = agent.build_project()
            print("MVP built and saved to workspace.")
            print(f"Result: {build_result}")

if __name__ == "__main__":
    main()
