import os
import subprocess
import sys

def main():
    # Default directory
    default_directory = "project_sample"

    if len(sys.argv) > 1:
        project_name = sys.argv[1]
    else:
        project_name = "default_project"

    full_path = os.path.join(default_directory, project_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    cli_directory = os.path.join(os.getcwd(), "gpt_engineer", "applications", "cli")
    os.chdir(cli_directory)

    command = ["python", "main.py", full_path]
    subprocess.run(command)

if __name__ == "__main__":
    main()
