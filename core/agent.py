import asyncio
from .llm import query_llm
from .maos_protocol import MAOSProtocol, AgentBlock
from .aider_integration import AiderIntegration, AiderConfig, MultiModalAiderOrchestrator
from utils.io import save_json, load_json

class EnhancedAgent:
    """
    Enhanced Agent with MAOS Protocol and Aider Integration
    
    Provides surgical prompt-to-product delivery with:
    - Multi-Agent Optimization System (MAOS)
    - Aider integration for code generation
    - Complete working codebase generation
    - Full deployment packages
    """
    
    def __init__(self, user_id: str, model: str = "gpt-4", api_key: str | None = None):
        self.user_id = user_id
        self.workspace_file = f"workspace/{user_id}.json"
        self.session = {"phases": {}, "current_phase": "idea", "maos_optimized": False}
        
        # Initialize MAOS protocol
        self.maos = MAOSProtocol(max_iterations=10, validation_threshold=0.85)
        
        # Initialize Aider integration
        aider_config = AiderConfig(
            model=model,
            api_key=api_key,
            workspace_dir=f"workspace/{user_id}",
            max_iterations=10,
            auto_commits=True
        )
        self.aider = AiderIntegration(aider_config)
        self.orchestrator = MultiModalAiderOrchestrator(aider_config)
        
        # Load or create session
        self._load()

    def _load(self):
        """Load session from file"""
        try:
            self.session = load_json(self.workspace_file)
        except FileNotFoundError:
            self._save()

    def _save(self):
        """Save session to file"""
        save_json(self.workspace_file, self.session)

    async def run_phase(self, input_text: str):
        """
        Run the current phase with MAOS optimization if applicable
        """
        phase = self.session["current_phase"]
        
        # Check if we should apply MAOS optimization
        if phase == "logic" and not self.session.get("maos_optimized", False):
            return await self._run_maos_optimized_phase(input_text)
        else:
            return await self._run_standard_phase(input_text)

    async def _run_standard_phase(self, input_text: str):
        """Run standard phase processing"""
        phase = self.session["current_phase"]
        prompt = f"[{phase.upper()} PHASE]\nInput: {input_text}"
        result = query_llm(prompt)
        self.session["phases"][phase] = result
        self._advance_phase()
        self._save()
        return result

    async def _run_maos_optimized_phase(self, input_text: str):
        """
        Run MAOS-optimized phase for complex multi-agent processing
        """
        print("🚀 Initializing MAOS Protocol for advanced optimization...")
        
        # Create agent blocks based on input
        agent_blocks = self._create_context_aware_agent_blocks(input_text)
        
        # Run MAOS optimization
        optimized_topology = await self.maos.optimize_complete_workflow(agent_blocks)
        
        # Execute the optimized workflow
        result = await self._execute_optimized_workflow(optimized_topology, input_text)
        
        # Store results
        self.session["phases"]["logic"] = result
        self.session["maos_optimized"] = True
        self.session["maos_score"] = optimized_topology.performance_score
        
        self._advance_phase()
        self._save()
        
        return result

    def _create_context_aware_agent_blocks(self, input_text: str) -> list:
        """Create agent blocks based on input context"""
        
        # Analyze input to determine needed agent types
        input_lower = input_text.lower()
        
        blocks = []
        
        # Always include predictor for analysis
        blocks.append(AgentBlock(
            "predictor", 
            f"Analyze this requirement: {input_text}\nProvide detailed technical predictions."
        ))
        
        # Add aggregator for synthesis
        blocks.append(AgentBlock(
            "aggregator",
            f"Combine multiple technical insights about: {input_text}"
        ))
        
        # Add reflector for quality assurance
        blocks.append(AgentBlock(
            "reflector",
            f"Review and improve the technical approach for: {input_text}"
        ))
        
        # Conditional blocks based on input
        if any(keyword in input_lower for keyword in ["web", "api", "service", "server"]):
            blocks.append(AgentBlock(
                "tool-use",
                f"Determine appropriate tools and frameworks for: {input_text}"
            ))
        
        if any(keyword in input_lower for keyword in ["complex", "multiple", "various"]):
            blocks.append(AgentBlock(
                "debate",
                f"Evaluate different approaches for: {input_text}"
            ))
        
        if len(input_text) > 500:  # Long input
            blocks.append(AgentBlock(
                "summarizer",
                f"Summarize key requirements from: {input_text}"
            ))
        
        return blocks

    async def _execute_optimized_workflow(self, topology, input_text: str):
        """Execute the optimized workflow topology"""
        
        # Simulate workflow execution
        # In a real implementation, this would execute the actual topology
        
        results = []
        for node in topology.nodes:
            # Execute each node with optimized prompts
            node_prompt = node.prompt_template.format(
                context=input_text,
                task=f"Phase: {self.session['current_phase']}",
                input=input_text
            )
            
            node_result = query_llm(node_prompt)
            results.append(f"[{node.agent_type.upper()}] {node_result}")
        
        # Aggregate results
        final_result = "\n\n".join(results)
        final_result += f"\n\n[MAOS OPTIMIZATION SCORE: {topology.performance_score:.3f}]"
        
        return final_result

    def _advance_phase(self):
        """Advance to next phase"""
        order = ["idea", "scope", "plan", "logic", "mvp"]
        i = order.index(self.session["current_phase"])
        if i + 1 < len(order):
            self.session["current_phase"] = order[i + 1]

    async def build_project(self):
        """
        Build complete project using Aider integration
        """
        print("🔨 Building complete project with Aider integration...")
        
        # Gather all phase information
        project_description = self._compile_project_description()
        
        # Create project using Aider
        result = await self.aider.create_project_from_prompt(
            project_description,
            f"{self.user_id}_project"
        )
        
        if result["success"]:
            print(f"✅ Project created successfully at: {result['project_path']}")
            
            # Apply additional refinements
            refinements = [
                "Add comprehensive error handling and logging",
                "Implement input validation and security measures",
                "Add configuration management and environment variables",
                "Create comprehensive test suite",
                "Optimize for production deployment"
            ]
            
            refinement_results = await self.aider.iterative_code_refinement(
                result["project_path"],
                refinements
            )
            
            print(f"✅ Applied {len(refinement_results)} refinements")
            
            # Create deployment package
            package_path = await self.aider.create_deployment_package(
                result["project_path"],
                package_format="zip",
                include_docs=True
            )
            
            print(f"📦 Deployment package created: {package_path}")
            
            # Store project info in session
            self.session["project_info"] = {
                "project_path": result["project_path"],
                "package_path": package_path,
                "refinements_applied": len(refinement_results),
                "project_structure": result["project_info"]
            }
            
            self._save()
            
            return {
                "success": True,
                "project_path": result["project_path"],
                "package_path": package_path,
                "message": "Complete working project built and packaged successfully"
            }
        else:
            print(f"❌ Project creation failed: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}

    def _compile_project_description(self) -> str:
        """Compile comprehensive project description from all phases"""
        
        description_parts = []
        
        # Add phase information
        for phase, content in self.session["phases"].items():
            description_parts.append(f"--- {phase.upper()} PHASE ---")
            description_parts.append(content)
            description_parts.append("")
        
        # Add MAOS optimization info if available
        if self.session.get("maos_optimized"):
            description_parts.append("--- MAOS OPTIMIZATION ---")
            description_parts.append(f"Optimization Score: {self.session.get('maos_score', 'N/A')}")
            description_parts.append("This project has been optimized using Multi-Agent Optimization System")
            description_parts.append("")
        
        # Add specific requirements
        description_parts.extend([
            "--- IMPLEMENTATION REQUIREMENTS ---",
            "1. Generate complete, working code - NO stubs or placeholders",
            "2. Include full file structure with all necessary files",
            "3. Add comprehensive documentation and setup instructions",
            "4. Ensure production-ready code with error handling",
            "5. Include all dependencies and configuration files",
            "6. Create working build/deployment scripts",
            "7. Add comprehensive test coverage",
            "8. Include README with step-by-step setup guide",
            ""
        ])
        
        return "\n".join(description_parts)

    async def create_complex_project(self, project_spec: dict):
        """
        Create complex project using multi-modal orchestration
        """
        print("🎯 Creating complex project with multi-modal orchestration...")
        
        result = await self.orchestrator.orchestrate_complex_project(
            project_spec,
            num_workers=3
        )
        
        if result["success"]:
            print("✅ Complex project integrated successfully")
            
            # Store results
            self.session["complex_project"] = result
            self._save()
            
            return result
        else:
            print("❌ Complex project creation failed")
            return result

# Legacy Agent class for backward compatibility
class Agent(EnhancedAgent):
    """Legacy Agent class - delegates to EnhancedAgent"""
    
    def __init__(self, user_id: str):
        super().__init__(user_id)
    
    def run_phase(self, input_text: str):
        """Legacy sync method - runs async version"""
        return asyncio.run(super().run_phase(input_text))
    
    def build_project(self):
        """Legacy sync method - runs async version"""
        return asyncio.run(super().build_project())
