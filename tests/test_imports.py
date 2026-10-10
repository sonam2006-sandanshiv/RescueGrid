import ast
import os
from pathlib import Path

def test_no_evaluation_imports_in_src():
    src_dir = Path("src")
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            assert not name.name.startswith("evaluation"), f"Illegal import '{name.name}' in {file_path}"
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            assert not node.module.startswith("evaluation"), f"Illegal from import '{node.module}' in {file_path}"
