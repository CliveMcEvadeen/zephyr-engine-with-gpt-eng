"""
Entrypoint for the CLI tool.

This module serves as the entry point for a command-line interface (CLI) tool.
It is designed to interact with the Llama language model.
The module provides functionality to:
- Load necessary environment variables,
- Configure various parameters for the AI interaction,
- Manage the generation or improvement of code projects.

Main Functionality
------------------
- Load environment variables required for Llama API interaction.
- Parse user-specified parameters for project configuration and AI behavior.
- Facilitate interaction with AI models, databases, and archival processes.

Parameters
----------
None

Notes
-----
- The `LLAMA_API_KEY` must be set in the environment or provided in a `.env` file within the working directory.
- The default project path is `projects/example`.
- When using the `azure_endpoint` parameter, provide the Azure OpenAI service endpoint URL.
"""

import difflib
import logging
import os
import sys

from pathlib import Path

import typer

from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from termcolor import colored

from llama_api.cli.cli_agent import CliAgent
from llama_api.cli.collect import collect_and_send_human_review
from llama_api.cli.file_selector import FileSelector
from llama_api.core.ai import AI, ClipboardAI
from llama_api.core.default.disk_execution_env import DiskExecutionEnv
from llama_api.core.default.disk_memory import DiskMemory
from llama_api.core.default.file_store import FileStore
from llama_api.core.default.paths import PREPROMPTS_PATH, memory_path
from llama_api.core.default.steps import (
    execute_entrypoint,
    gen_code,
    handle_improve_mode,
    improve_fn as improve_fn,
)
from llama_api.core.files_dict import FilesDict
from llama_api.core.git import stage_uncommitted_to_git
from llama_api.core.preprompts_holder import PrepromptsHolder
from llama_api.core.prompt import Prompt
from llama_api.tools.custom_steps import clarified_gen, lite_gen, self_heal

app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}
)  # creates a CLI app


def load_env_if_needed():
    """
    Load environment variables if the LLAMA_API_KEY is not already set.

    This function checks if the LLAMA_API_KEY environment variable is set,
    and if not, it attempts to load it from a .env file in the current working
    directory. It then sets the llama.api_key for use in the application.
    """
    if os.getenv("LLAMA_API_KEY") is None:
        load_dotenv()
    if os.getenv("LLAMA_API_KEY") is None:
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

    os.environ["LLAMA_API_KEY"] = os.getenv("LLAMA_API_KEY")


def concatenate_paths(base_path, sub_path):
    # Compute the relative path from base_path to sub_path
    relative_path = os.path.relpath(sub_path, base_path)

    # If the relative path is not in the parent directory, use the original sub_path
    if not relative_path.startswith(".."):
        return sub_path

    # Otherwise, concatenate base_path and sub_path
    return os.path.normpath(os.path.join(base_path, sub_path))


def load_prompt(
    input_repo: DiskMemory,
    improve_mode: bool,
    prompt_file: str,
    image_directory: str,
    entrypoint_prompt_file: str = "",
) -> Prompt:
    """
    Load or request a prompt from the user based on the mode.

    Parameters
    ----------
    input_repo : DiskMemory
        The disk memory object where prompts and other data are stored.
    improve_mode : bool
        Flag indicating whether the application is in improve mode.

    Returns
    -------
    str
        The loaded or inputted prompt.
    """

    if os.path.isdir(prompt_file):
        raise ValueError(
            f"The path to the prompt, {prompt_file}, already exists as a directory. No prompt can be read from it. Please specify a prompt file using --prompt_file"
        )
    prompt_str = input_repo.get(prompt_file)
    if prompt_str:
        print(colored("Using prompt from file:", "green"), prompt_file)
        print(prompt_str)
    else:
        if not improve_mode:
            prompt_str = input(colored(
                "\nWhat application do you want llama-engineer to generate?\n"
            , 'green'))
        else:
            prompt_str = input("\nHow do you want to improve the application?\n")

    if entrypoint_prompt_file == "":
        entrypoint_prompt = ""
    else:
        full_entrypoint_prompt_file = concatenate_paths(
            input_repo.path, entrypoint_prompt_file
        )
        if os.path.isfile(full_entrypoint_prompt_file):
            entrypoint_prompt = input_repo.get(full_entrypoint_prompt_file)

        else:
            raise ValueError("The provided file at --entrypoint-prompt does not exist")

    if image_directory == "":
        return Prompt(prompt_str, entrypoint_prompt=entrypoint_prompt)

    full_image_directory = concatenate_paths(input_repo.path, image_directory)
    if os.path.isdir(full_image_directory):
        if len(os.listdir(full_image_directory)) == 0:
            raise ValueError("The provided --image_directory is empty.")
        image_repo = DiskMemory(full_image_directory)
        return Prompt(
            prompt_str,
            image_repo.get(".").to_dict(),
            entrypoint_prompt=entrypoint_prompt,
        )
    else:
        raise ValueError("The provided --image_directory is not a directory.")


def get_preprompts_path(use_custom_preprompts: bool, input_path: Path) -> Path:
    """
    Get the path to the preprompts, using custom ones if specified.

    Parameters
    ----------
    use_custom_preprompts : bool
        Flag indicating whether to use custom preprompts.
    input_path : Path
        The path to the project directory.

    Returns
    -------
    Path
        The path to the directory containing the preprompts.
    """
    original_preprompts_path = PREPROMPTS_PATH
    if not use_custom_preprompts:
        return original_preprompts_path

    custom_preprompts_path = input_path / "preprompts"
    if not custom_preprompts_path.exists():
        custom_preprompts_path.mkdir()

    for file in original_preprompts_path.glob("*"):
        if not (custom_preprompts_path / file.name).exists():
            (custom_preprompts_path / file.name).write_text(file.read_text())
    return custom_preprompts_path


def compare(f1: FilesDict, f2: FilesDict):
    def colored_diff(s1, s2):
        lines1 = s1.splitlines()
        lines2 = s2.splitlines()

        diff = difflib.unified_diff(lines1, lines2, lineterm="")

        RED = "\033[38;5;202m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        colored_lines = []
        for line in diff:
            if line.startswith("+"):
                colored_lines.append(GREEN + line + RESET)
            elif line.startswith("-"):
                colored_lines.append(RED + line + RESET)
            else:
                colored_lines.append(line)

        return "\n".join(colored_lines)

    for file in sorted(set(f1) | set(f2)):
        diff = colored_diff(f1.get(file, ""), f2.get(file, ""))
        if diff:
            print(f"Changes to {file}:")
            print(diff)


def prompt_yesno() -> bool:
    TERM_CHOICES = colored("y", "green") + "/" + colored("n", "red") + " "
    while True:
        response = input(TERM_CHOICES).strip().lower()
        if response in ["y", "yes"]:
            return True
        if response in ["n", "no"]:
            break
        print("Please respond with 'y' or 'n'")


@app.command(
    help="""
        Llama-engineer lets you:

        \b
        - Specify a software in natural language
        - Sit back and watch as an AI writes and executes the code
        - Ask the AI to implement improvements
    """
)
def main(
    project_path: str = typer.Argument(".", help="path"),
    model: str = typer.Option(
        os.environ.get("MODEL_NAME", "gemini-1.5-pro"), "--model", "-m", help="model id string"
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-t",
        help="Controls randomness: lower values for more focused, deterministic outputs",
    ),
    improve_mode: bool = typer.Option(
        False,
        "--improve",
        "-i",
        help="Improve an existing project by modifying the files.",
    ),
    lite_mode: bool = typer.Option(
        False,
        "--lite",
        "-l",
        help="Lite mode: run a generation using only the main prompt.",
    ),
    clarify_mode: bool = typer.Option(
        False,
        "--clarify",
        "-c",
        help="Clarify mode - discuss specification with AI before implementation.",
    ),
    self_heal_mode: bool = typer.Option(
        False,
        "--self-heal",
        "-sh",
        help="Self-heal mode - fix the code by itself when it fails.",
    ),
    azure_endpoint: str = typer.Option(
        "",
        "--azure",
        "-a",
        help="""Endpoint for your Azure OpenAI Service (https://xx.openai.azure.com)
        Note: set environment variables LLAMA_API_KEY and AZURE_API_VERSION""",
    ),
    human_review_mode: bool = typer.Option(
        False,
        "--human-review",
        "-hr",
        help="Human review mode - send generated content for human review before making changes.",
    ),
    custom_preprompts: bool = typer.Option(
        False,
        "--custom-preprompts",
        "-cp",
        help="Use custom preprompts from the project's preprompts directory.",
    ),
    image_directory: str = typer.Option(
        "",
        "--image-directory",
        "-img",
        help="Directory containing images relevant to the project.",
    ),
    prompt_file: str = typer.Option(
        ".prompt",
        "--prompt-file",
        "-pf",
        help="File containing the prompt for the AI to generate or improve the application.",
    ),
    entrypoint_prompt_file: str = typer.Option(
        "",
        "--entrypoint-prompt",
        "-ep",
        help="File containing the entrypoint prompt for the AI to start with.",
    ),
):
    """
    The main function to handle project initialization, AI configuration, and task execution.

    Parameters
    ----------
    project_path : str, optional
        The path to the project directory, by default ".".
    model : str, optional
        The model ID string to be used for AI interaction, by default the environment variable `MODEL_NAME` or "gemini-1.5-pro".
    temperature : float, optional
        The temperature setting for the AI model, by default 0.1.
    improve_mode : bool, optional
        Flag to indicate if the project is in improvement mode, by default False.
    lite_mode : bool, optional
        Flag to indicate if lite mode should be used, by default False.
    clarify_mode : bool, optional
        Flag to indicate if clarify mode should be used, by default False.
    self_heal_mode : bool, optional
        Flag to indicate if self-heal mode should be used, by default False.
    azure_endpoint : str, optional
        The endpoint for the Azure OpenAI service, by default "".
    human_review_mode : bool, optional
        Flag to indicate if human review mode should be used, by default False.
    custom_preprompts : bool, optional
        Flag to indicate if custom preprompts should be used, by default False.
    image_directory : str, optional
        Directory containing images relevant to the project, by default "".
    prompt_file : str, optional
        File containing the prompt for the AI to generate or improve the application, by default ".prompt".
    entrypoint_prompt_file : str, optional
        File containing the entrypoint prompt for the AI to start with, by default "".
    """

    load_env_if_needed()

    set_llm_cache(SQLiteCache("llm_cache.db"))
    if azure_endpoint:
        print(
            f"Using Azure OpenAI Service endpoint: {azure_endpoint}"
        )

    input_path = Path(project_path)
    input_repo = DiskMemory(input_path)
    input_files = FilesDict({key: input_repo.get(key) for key in input_repo.keys()})

    agent = CliAgent(model=model, temperature=temperature, azure_endpoint=azure_endpoint)

    preprompts_path = get_preprompts_path(custom_preprompts, input_path)
    preprompts = PrepromptsHolder(preprompts_path)

    ai = AI(
        agent,
        input_path,
        preprompts,
        image_directory=image_directory,
    )

    if improve_mode:
        if lite_mode:
            raise ValueError("--lite and --improve cannot be used together")
        if clarify_mode:
            raise ValueError("--clarify and --improve cannot be used together")
        improve_fn(
            input_repo,
            input_files,
            agent,
            preprompts,
            self_heal_mode=self_heal_mode,
        )
    elif clarify_mode:
        clarified_gen(
            input_repo,
            input_files,
            agent,
            preprompts,
            image_directory=image_directory,
            human_review_mode=human_review_mode,
        )
    elif lite_mode:
        lite_gen(
            input_repo,
            input_files,
            agent,
            preprompts,
            image_directory=image_directory,
            human_review_mode=human_review_mode,
        )
    else:
        prompt = load_prompt(
            input_repo,
            improve_mode,
            prompt_file,
            image_directory,
            entrypoint_prompt_file,
        )
        output_repo = DiskExecutionEnv(
            agent,
            input_path,
            preprompts,
            self_heal_mode=self_heal_mode,
        ).execute(prompt)

        print(f"Generated code to {output_repo.path}")
        stage_uncommitted_to_git(input_repo.path)

        compare(input_files, FilesDict({key: output_repo.get(key) for key in output_repo.keys()}))
        if human_review_mode:
            print("Submit this change for human review?")
            if prompt_yesno():
                collect_and_send_human_review(output_repo)
            else:
                print("Skipping human review.")

if __name__ == "__main__":
    app()
