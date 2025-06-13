"""
Aider Integration Module for Agent Foundry

This module integrates aider-chat and ai-cli for enhanced code generation capabilities.
Provides multi-iterative command running capabilities and surgical code modifications.
"""

import asyncio
import os
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AiderConfig:
    """Configuration for aider integration"""
    model: str = "gpt-4"
    api_key: Optional[str] = None
    max_iterations: int = 10
    auto_commits: bool = True
    verbose: bool = True
    workspace_dir: str = "./workspace"

@dataclass
class CommandResult:
    """Result of a command execution"""
    success: bool
    output: str
    error: str
    return_code: int
    duration: float

class AiderIntegration:
    """
    Aider Integration for surgical code generation and modification
    
    Provides capabilities for:
    - Multi-iterative code generation
    - Surgical prompt-to-code modifications
    - AI-CLI command orchestration
    - Complete codebase generation with full file structures
    """
    
    def __init__(self, config: AiderConfig):
        self.config = config
        self.workspace_path = Path(config.workspace_dir)
        self.current_session = None
        self.command_history = []
        
        # Ensure workspace exists
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
    async def create_project_from_prompt(self, prompt: str, project_name: str) -> Dict[str, Any]:
        """
        Create a complete project from a natural language prompt
        
        Args:
            prompt: Natural language description of the project
            project_name: Name for the project directory
            
        Returns:
            Dictionary with project creation results
        """
        logger.info(f"Creating project '{project_name}' from prompt")
        
        project_path = self.workspace_path / project_name
        project_path.mkdir(exist_ok=True)
        
        # Phase 1: Project structure planning
        structure_prompt = f"""
Please create a complete project structure for: {prompt}

Requirements:
1. Generate a complete directory structure
2. Create all necessary files with full implementations
3. Include package.json, requirements.txt, or equivalent dependency files
4. Add README.md with setup instructions
5. Include any configuration files needed
6. Generate working code - no stubs or placeholders
7. Ensure the project is fully functional and deployable

Create the project in the current directory.
"""
        
        # Execute aider for project creation
        result = await self._run_aider_command(
            structure_prompt,
            working_dir=str(project_path)
        )
        
        if not result.success:
            logger.error(f"Failed to create project structure: {result.error}")
            return {"success": False, "error": result.error}
        
        # Phase 2: Iterative refinement
        refinement_prompts = [
            "Review the generated code and add any missing error handling",
            "Add comprehensive logging throughout the application",
            "Ensure all dependencies are properly specified",
            "Add input validation and security considerations",
            "Optimize the code for production deployment"
        ]
        
        for i, refinement in enumerate(refinement_prompts):
            logger.info(f"Applying refinement {i+1}/{len(refinement_prompts)}")
            
            result = await self._run_aider_command(
                refinement,
                working_dir=str(project_path)
            )
            
            if not result.success:
                logger.warning(f"Refinement {i+1} failed: {result.error}")
        
        # Phase 3: Documentation and testing
        doc_prompt = """
Generate comprehensive documentation:
1. Update README.md with detailed setup and usage instructions
2. Add inline code comments and docstrings
3. Create API documentation if applicable
4. Add example usage files
5. Include troubleshooting section
"""
        
        await self._run_aider_command(doc_prompt, working_dir=str(project_path))
        
        # Generate project summary
        project_info = await self._analyze_project_structure(project_path)
        
        return {
            "success": True,
            "project_path": str(project_path),
            "project_info": project_info,
            "command_history": self.command_history[-10:]  # Last 10 commands
        }
    
    async def _run_aider_command(self, prompt: str, working_dir: str) -> CommandResult:
        """Execute an aider command with the given prompt"""
        
        # Prepare aider command
        cmd = [
            "aider",
            "--message", prompt,
            "--model", self.config.model
        ]
        
        if self.config.auto_commits:
            cmd.append("--auto-commits")
        
        if self.config.verbose:
            cmd.append("--verbose")
        
        if self.config.api_key:
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = self.config.api_key
        else:
            env = None
        
        logger.info(f"Running aider command in {working_dir}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Run aider command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            duration = asyncio.get_event_loop().time() - start_time
            
            result = CommandResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8') if stdout else "",
                error=stderr.decode('utf-8') if stderr else "",
                return_code=process.returncode,
                duration=duration
            )
            
            # Store command in history
            self.command_history.append({
                "prompt": prompt,
                "working_dir": working_dir,
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute aider command: {str(e)}")
            return CommandResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                duration=asyncio.get_event_loop().time() - start_time
            )
    
    async def run_ai_cli_command(self, command: str, working_dir: str) -> CommandResult:
        """Execute an ai-cli command for additional AI capabilities"""
        
        cmd = ["ai", command]
        
        logger.info(f"Running ai-cli command: {command}")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            duration = asyncio.get_event_loop().time() - start_time
            
            result = CommandResult(
                success=process.returncode == 0,
                output=stdout.decode('utf-8') if stdout else "",
                error=stderr.decode('utf-8') if stderr else "",
                return_code=process.returncode,
                duration=duration
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute ai-cli command: {str(e)}")
            return CommandResult(
                success=False,
                output="",
                error=str(e),
                return_code=-1,
                duration=asyncio.get_event_loop().time() - start_time
            )
    
    async def iterative_code_refinement(self, 
                                      project_path: str, 
                                      refinement_instructions: List[str],
                                      max_iterations: Optional[int] = None) -> List[CommandResult]:
        """
        Apply iterative refinements to a codebase
        
        Args:
            project_path: Path to the project directory
            refinement_instructions: List of refinement prompts
            max_iterations: Maximum number of iterations (defaults to config)
            
        Returns:
            List of command results
        """
        
        max_iter = max_iterations or self.config.max_iterations
        results = []
        
        for i, instruction in enumerate(refinement_instructions[:max_iter]):
            logger.info(f"Applying refinement {i+1}/{min(len(refinement_instructions), max_iter)}")
            
            # Create context-aware prompt
            context_prompt = f"""
Iteration {i+1} of code refinement:

Current task: {instruction}

Guidelines:
- Review the existing code thoroughly
- Make surgical modifications where needed
- Ensure all changes maintain code quality
- Test compatibility with existing functionality
- Document any significant changes

Apply the refinement now:
"""
            
            result = await self._run_aider_command(context_prompt, project_path)
            results.append(result)
            
            # Short pause between iterations
            await asyncio.sleep(1)
        
        return results
    
    async def _analyze_project_structure(self, project_path: Path) -> Dict[str, Any]:
        """Analyze the generated project structure"""
        
        structure = {
            "files": [],
            "directories": [],
            "total_files": 0,
            "total_size": 0,
            "languages": set(),
            "dependencies": []
        }
        
        try:
            for item in project_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(project_path)
                    file_info = {
                        "path": str(relative_path),
                        "size": item.stat().st_size,
                        "extension": item.suffix
                    }
                    structure["files"].append(file_info)
                    structure["total_files"] += 1
                    structure["total_size"] += file_info["size"]
                    
                    # Detect language
                    if item.suffix:
                        structure["languages"].add(item.suffix)
                    
                    # Check for dependency files
                    if item.name in ["package.json", "requirements.txt", "Gemfile", "pom.xml", "go.mod"]:
                        try:
                            with open(item, 'r') as f:
                                content = f.read()
                                structure["dependencies"].append({
                                    "file": str(relative_path),
                                    "content": content[:500]  # First 500 chars
                                })
                        except Exception:
                            pass
                            
                elif item.is_dir():
                    structure["directories"].append(str(item.relative_to(project_path)))
            
            structure["languages"] = list(structure["languages"])
            
        except Exception as e:
            logger.error(f"Failed to analyze project structure: {str(e)}")
        
        return structure
    
    async def create_deployment_package(self, 
                                      project_path: str, 
                                      package_format: str = "zip",
                                      include_docs: bool = True) -> str:
        """
        Create a deployment package for the generated project
        
        Args:
            project_path: Path to the project directory
            package_format: Format for the package (zip, tar.gz, etc.)
            include_docs: Whether to include documentation
            
        Returns:
            Path to the created package
        """
        
        project_path_obj = Path(project_path)
        package_name = f"{project_path_obj.name}_deployment"
        
        if package_format.lower() == "zip":
            package_file = f"{package_name}.zip"
            package_path = self.workspace_path / package_file
            
            # Create zip package
            shutil.make_archive(
                str(self.workspace_path / package_name),
                'zip',
                project_path
            )
            
        elif package_format.lower() in ["tar.gz", "tgz"]:
            package_file = f"{package_name}.tar.gz"
            package_path = self.workspace_path / package_file
            
            # Create tar.gz package
            shutil.make_archive(
                str(self.workspace_path / package_name),
                'gztar',
                project_path
            )
        
        else:
            raise ValueError(f"Unsupported package format: {package_format}")
        
        # Add deployment documentation if requested
        if include_docs:
            await self._add_deployment_docs(project_path_obj)
        
        logger.info(f"Created deployment package: {package_path}")
        return str(package_path)
    
    async def _add_deployment_docs(self, project_path: Path):
        """Add deployment-specific documentation"""
        
        deployment_prompt = """
Create comprehensive deployment documentation:

1. Create DEPLOYMENT.md with:
   - System requirements
   - Installation steps
   - Configuration instructions
   - Environment setup
   - Deployment commands
   - Troubleshooting guide

2. Add docker files if applicable:
   - Dockerfile
   - docker-compose.yml
   - .dockerignore

3. Create deployment scripts:
   - deploy.sh (for Unix systems)
   - deploy.bat (for Windows)
   - Environment configuration templates

4. Add monitoring and health check endpoints if this is a web service

Make sure all deployment artifacts are production-ready.
"""
        
        await self._run_aider_command(deployment_prompt, str(project_path))

class MultiModalAiderOrchestrator:
    """
    Orchestrates multiple aider instances for complex project generation
    """
    
    def __init__(self, config: AiderConfig):
        self.config = config
        self.aider_instances = []
        self.coordination_queue = asyncio.Queue()
        
    async def orchestrate_complex_project(self, 
                                        project_spec: Dict[str, Any],
                                        num_workers: int = 3) -> Dict[str, Any]:
        """
        Orchestrate multiple aider instances for complex project creation
        
        Args:
            project_spec: Detailed project specification
            num_workers: Number of parallel aider workers
            
        Returns:
            Orchestration results
        """
        
        # Break down project into components
        components = self._decompose_project(project_spec)
        
        # Create worker instances
        workers = []
        for i in range(num_workers):
            worker = AiderIntegration(self.config)
            workers.append(worker)
        
        # Distribute work among workers
        tasks = []
        for i, component in enumerate(components):
            worker = workers[i % num_workers]
            task = asyncio.create_task(
                worker.create_project_from_prompt(
                    component["prompt"],
                    component["name"]
                )
            )
            tasks.append(task)
        
        # Wait for all components to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Integrate results
        integration_result = await self._integrate_components(results, project_spec)
        
        return integration_result
    
    def _decompose_project(self, project_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose a complex project into manageable components"""
        
        # This is a simplified decomposition - in practice, this would be much more sophisticated
        components = []
        
        # Extract main components based on project type
        if "web" in project_spec.get("type", "").lower():
            components.extend([
                {"name": "frontend", "prompt": f"Create frontend for: {project_spec['description']}"},
                {"name": "backend", "prompt": f"Create backend API for: {project_spec['description']}"},
                {"name": "database", "prompt": f"Create database schema for: {project_spec['description']}"}
            ])
        elif "api" in project_spec.get("type", "").lower():
            components.extend([
                {"name": "api_core", "prompt": f"Create REST API for: {project_spec['description']}"},
                {"name": "documentation", "prompt": f"Create API documentation for: {project_spec['description']}"},
                {"name": "tests", "prompt": f"Create comprehensive tests for: {project_spec['description']}"}
            ])
        else:
            # Generic decomposition
            components.append({
                "name": "main_project",
                "prompt": project_spec["description"]
            })
        
        return components
    
    async def _integrate_components(self, 
                                  component_results: List[Any], 
                                  project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate multiple component results into a unified project"""
        
        successful_components = [r for r in component_results if isinstance(r, dict) and r.get("success")]
        
        if not successful_components:
            return {"success": False, "error": "No components were successfully created"}
        
        # Create integration workspace
        integration_path = Path(self.config.workspace_dir) / f"{project_spec.get('name', 'integrated_project')}"
        integration_path.mkdir(exist_ok=True)
        
        # Copy all components to integration workspace
        for component in successful_components:
            component_path = Path(component["project_path"])
            if component_path.exists():
                # Copy component files to integration workspace
                for item in component_path.rglob("*"):
                    if item.is_file():
                        relative_path = item.relative_to(component_path)
                        target_path = integration_path / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target_path)
        
        # Create integration aider instance
        integrator = AiderIntegration(self.config)
        
        # Run integration prompt
        integration_prompt = f"""
Integrate the following components into a unified project:

Project Goal: {project_spec.get('description', 'Integrated application')}

Components included: {[c.get('project_info', {}).get('files', []) for c in successful_components]}

Tasks:
1. Ensure all components work together seamlessly
2. Create a unified build system
3. Add integration tests
4. Update documentation to reflect the integrated system
5. Resolve any conflicts between components
6. Add a master configuration system
7. Create unified deployment scripts

Make this a production-ready, fully integrated system.
"""
        
        integration_result = await integrator._run_aider_command(
            integration_prompt,
            str(integration_path)
        )
        
        return {
            "success": integration_result.success,
            "integrated_project_path": str(integration_path),
            "component_count": len(successful_components),
            "integration_result": integration_result
        }

# Factory function for easy integration
def create_aider_integration(model: str = "gpt-4", 
                           workspace_dir: str = "./workspace",
                           api_key: Optional[str] = None) -> AiderIntegration:
    """Factory function to create AiderIntegration instance"""
    
    config = AiderConfig(
        model=model,
        workspace_dir=workspace_dir,
        api_key=api_key
    )
    
    return AiderIntegration(config)

async def main():
    """Test the aider integration"""
    
    # Create aider integration
    aider = create_aider_integration()
    
    # Test project creation
    result = await aider.create_project_from_prompt(
        "Create a simple REST API for managing tasks with CRUD operations",
        "task_api"
    )
    
    print(f"Project creation result: {result}")
    
    if result["success"]:
        # Create deployment package
        package_path = await aider.create_deployment_package(
            result["project_path"],
            package_format="zip"
        )
        print(f"Deployment package created: {package_path}")

if __name__ == "__main__":
    asyncio.run(main())
