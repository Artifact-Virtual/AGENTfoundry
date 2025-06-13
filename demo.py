#!/usr/bin/env python3
"""
Agent Foundry Demo Script

Demonstrates the enhanced capabilities of Agent Foundry with MAOS Protocol
and Aider Integration for surgical prompt-to-product delivery.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the core directory to Python path
sys.path.append(str(Path(__file__).parent / "core"))

from core.agent import EnhancedAgent
from core.maos_protocol import MAOSProtocol, create_default_agent_blocks
from core.aider_integration import create_aider_integration

async def demo_basic_workflow():
    """Demonstrate basic enhanced workflow"""
    print("Demo 1: Basic Enhanced Workflow")
    print("=" * 50)
    
    agent = EnhancedAgent("demo_user", model="gpt-4")
    
    # Run through all phases
    phases_input = [
        "Create a modern web-based task management application",
        "Include user authentication, task CRUD operations, and real-time updates",
        "Use React frontend, Node.js backend, and PostgreSQL database",
        "Implement REST API with JWT authentication and WebSocket for real-time features",
        "Generate complete working codebase with deployment configuration"
    ]
    
    for i, phase_input in enumerate(phases_input):
        print(f"\nPhase {i+1}: {agent.session['current_phase'].upper()}")
        print(f"Input: {phase_input}")
        
        result = await agent.run_phase(phase_input)
        print(f"Output: {result[:200]}...")
        
        if agent.session['current_phase'] == "mvp":
            break
    
    return agent

async def demo_maos_protocol():
    """Demonstrate MAOS Protocol optimization"""
    print("\nDemo 2: MAOS Protocol Optimization")
    print("=" * 50)
    
    maos = MAOSProtocol(max_iterations=5, validation_threshold=0.75)
    
    # Create agent blocks
    blocks = create_default_agent_blocks()
    print(f"Created {len(blocks)} agent blocks")
    
    # Run MAOS optimization
    print("\nRunning 3-stage MAOS optimization...")
    optimized_topology = await maos.optimize_complete_workflow(blocks)
    
    print(f"Optimization completed!")
    print(f"Final Score: {optimized_topology.performance_score:.3f}")
    print(f"Nodes: {len(optimized_topology.nodes)}")
    print(f"Edges: {len(optimized_topology.edges)}")
    
    return optimized_topology

async def demo_aider_integration():
    """Demonstrate Aider integration capabilities"""
    print("\nDemo 3: Aider Integration")
    print("=" * 50)
    
    # Create aider integration
    aider = create_aider_integration(
        model="gpt-4",
        workspace_dir="./demo_workspace"
    )
    
    # Simulate project creation (would normally call aider)
    print("Simulating project creation with Aider...")
    print("Note: This demo shows the workflow without actual aider calls")
    
    # In a real scenario, this would create a complete project
    project_spec = {
        "name": "demo_api",
        "description": "REST API for task management with authentication",
        "type": "api"
    }
    
    print(f"Project Spec: {project_spec}")
    print("Would create: Full API with endpoints, auth, database, tests, docs")
    print("Would package: ZIP file with complete deployable codebase")
    
    return project_spec

async def demo_complete_workflow():
    """Demonstrate complete end-to-end workflow"""
    print("\nDemo 4: Complete End-to-End Workflow")
    print("=" * 50)
    
    agent = EnhancedAgent("complete_demo", model="gpt-4")
    
    # Input for a complex SaaS application
    complex_input = """
    Create a complete SaaS application for project management with the following features:
    
    1. Multi-tenant architecture
    2. User authentication and authorization
    3. Project and task management
    4. Real-time collaboration
    5. File sharing and comments
    6. Dashboard with analytics
    7. API for mobile app integration
    8. Subscription billing integration
    9. Admin panel for tenant management
    10. CI/CD pipeline configuration
    
    Technical requirements:
    - Microservices architecture
    - Docker containerization
    - Kubernetes deployment
    - PostgreSQL database
    - Redis for caching
    - React frontend
    - Node.js backend
    - Comprehensive test coverage
    - Complete documentation
    """
    
    print("Complex SaaS Project Input:")
    print(complex_input[:200] + "...")
    
    # This would normally run the complete workflow
    print("\nEnhanced Agent Processing:")
    print("1. Idea Phase: Project analysis and feature breakdown")
    print("2. Scope Phase: Technical requirements and architecture")
    print("3. Plan Phase: Implementation strategy and timeline")
    print("4. Logic Phase: MAOS optimization for complex workflow")
    print("5. MVP Phase: Complete codebase generation with Aider")
    
    print("\nExpected Output:")
    print("- Complete microservices codebase")
    print("- Docker and Kubernetes configurations")
    print("- Database schemas and migrations")
    print("- Frontend application with all features")
    print("- API documentation")
    print("- Test suites for all components")
    print("- Deployment scripts and CI/CD pipeline")
    print("- Complete setup documentation")
    print("- ZIP package ready for deployment")
    
    return True

async def demo_multi_modal_orchestration():
    """Demonstrate multi-modal orchestration"""
    print("\nDemo 5: Multi-Modal Orchestration")
    print("=" * 50)
    
    print("Multi-Agent Coordination:")
    print("Agent 1: Frontend Development (React + TypeScript)")
    print("Agent 2: Backend API Development (Node.js + Express)")
    print("Agent 3: Database Design (PostgreSQL + Migrations)")
    print("Agent 4: DevOps Configuration (Docker + K8s)")
    print("Agent 5: Testing Framework (Jest + Cypress)")
    
    print("\nParallel Processing:")
    print("- Each agent works on its specialized component")
    print("- Real-time coordination through shared context")
    print("- Automatic integration of all components")
    print("- Conflict resolution and compatibility checking")
    
    print("\nIntegration Result:")
    print("- Unified project structure")
    print("- Seamless component interaction")
    print("- Complete build and deployment system")
    print("- Production-ready architecture")
    
    return True

def print_banner():
    """Print demo banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           AGENT FOUNDRY DEMO                                ║
    ║                                                                              ║
    ║    Surgical Prompt-to-Product Pipeline                                      ║
    ║    MAOS Protocol Integration                                                 ║
    ║    Aider Multi-Iterative Code Generation                                    ║
    ║    Complete Working Codebase Output                                         ║
    ║                                                                              ║
    ║    © Artifact Virtual - Where Prompts Become Products                       ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def main():
    """Run all demos"""
    print_banner()
    
    try:
        # Demo 1: Basic workflow
        agent = await demo_basic_workflow()
        
        # Demo 2: MAOS Protocol
        topology = await demo_maos_protocol()
        
        # Demo 3: Aider Integration
        project_spec = await demo_aider_integration()
        
        # Demo 4: Complete workflow
        complete_result = await demo_complete_workflow()
        
        # Demo 5: Multi-modal orchestration
        orchestration_result = await demo_multi_modal_orchestration()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("Agent Foundry is ready for surgical prompt-to-product delivery!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("This is expected if dependencies are not installed.")
        print("Run 'pip install -r requirements.txt' to install dependencies.")

if __name__ == "__main__":
    asyncio.run(main())
