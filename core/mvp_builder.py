import os

def build_mvp(scope, logic):
    os.makedirs("workspace/mvp", exist_ok=True)
    main_code = f"# MVP based on:\n# Scope: {scope}\n# Logic: {logic}\n\nprint('Hello from MVP!')\n"
    with open("workspace/mvp/main.py", "w") as f:
        f.write(main_code)
    return "MVP built successfully."
