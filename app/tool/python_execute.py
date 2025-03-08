import threading
from typing import Dict

from app.tool.base import BaseTool
from app.logger import logger

class PythonExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""

    name: str = "python_execute"
    description: str = "Executes Python code string. Note: Only print outputs are visible, function return values are not captured. Use print statements to see results."
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
        },
        "required": ["code"],
    }

    def _format_code(self, code: str) -> str:
        """Format the code string to ensure proper line breaks and indentation."""
        # First, normalize line endings
        code = code.replace('\\n', '\n')
        
        # Split the code into lines
        lines = code.split('\n')
        
        # Process imports - split multiple imports into separate lines
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('import ') and ' from ' in line:
                # Handle "import X from Y" format
                parts = line.split(' from ')
                formatted_lines.append(f"from {parts[1]} import {parts[0].replace('import ', '')}")
            elif line.startswith('import ') and ',' in line:
                # Handle multiple imports
                imports = [imp.strip() for imp in line.split('import ')[1].split(',')]
                for imp in imports:
                    formatted_lines.append(f"import {imp}")
            elif line.startswith('from ') and ',' in line and ' import ' in line:
                # Handle multiple imports from same module
                module = line.split(' import ')[0]
                imports = [imp.strip() for imp in line.split(' import ')[1].split(',')]
                formatted_lines.append(f"{module} import {', '.join(imports)}")
            else:
                formatted_lines.append(line)
        
        # Remove empty lines at start and end
        while formatted_lines and not formatted_lines[0].strip():
            formatted_lines.pop(0)
        while formatted_lines and not formatted_lines[-1].strip():
            formatted_lines.pop()
            
        # Find minimum indentation (excluding import statements)
        min_indent = float('inf')
        for line in formatted_lines:
            if line.strip() and not line.strip().startswith(('import ', 'from ')):
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent) if indent < min_indent else min_indent
        if min_indent == float('inf'):
            min_indent = 0
            
        # Remove common indentation and join lines
        result_lines = []
        for line in formatted_lines:
            if line.strip().startswith(('import ', 'from ')):
                # Don't adjust indentation for import statements
                result_lines.append(line.strip())
            else:
                # Adjust indentation for other lines
                result_lines.append(line[min_indent:] if line.strip() else '')
        
        return '\n'.join(result_lines)

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        Executes the provided Python code with a timeout.

        Args:
            code (str): The Python code to execute.
            timeout (int): Execution timeout in seconds.

        Returns:
            Dict: Contains 'output' with execution output or error message and 'success' status.
        """
        result = {"observation": "", "success": True}
        
        # Format the code
        formatted_code = self._format_code(code)
        logger.info(f"🔧 Executing code:\n{formatted_code}")

        def run_code():
            try:
                # Set up a safe execution environment with basic modules
                safe_globals = {
                    "__builtins__": dict(__builtins__),
                    "print": print,  # Ensure print is available
                }

                # Import basic required modules
                import sys
                from io import StringIO
                safe_globals.update({'sys': sys})

                # Check for required imports in the code
                required_imports = []
                for line in formatted_code.split('\n'):
                    if line.strip().startswith(('import ', 'from ')):
                        required_imports.append(line.strip())

                # Try to import required modules
                local_dict = {}
                for import_stmt in required_imports:
                    try:
                        exec(import_stmt, safe_globals, local_dict)
                    except ImportError as e:
                        raise ImportError(f"Required module not found: {str(e)}. Please install missing dependencies.")
                
                # Update safe_globals with successfully imported modules
                safe_globals.update(local_dict)

                # Capture stdout
                output_buffer = StringIO()
                sys.stdout = output_buffer

                # Execute the formatted code
                exec(formatted_code, safe_globals, {})

                # Restore stdout and get output
                sys.stdout = sys.__stdout__
                result["observation"] = output_buffer.getvalue()

            except Exception as e:
                result["observation"] = str(e)
                result["success"] = False
                logger.error(f"Error executing code: {str(e)}")

        thread = threading.Thread(target=run_code)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            return {
                "observation": f"Execution timeout after {timeout} seconds",
                "success": False,
            }

        return result
