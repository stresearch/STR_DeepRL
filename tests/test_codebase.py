import subprocess
import os
import ast
import glob


ALLOWED_FUNCTIONS_IN_CLASS = {
    "CompetativeRedBlueDoorEnvV3": ["ctce_step", "dtde_step", "_handle_steps"],
}


def get_functions_in_classes_from_ast(tree):
    functions_in_classes = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            functions_in_class = []
            for class_node in ast.iter_child_nodes(node):
                if isinstance(class_node, ast.FunctionDef):
                    functions_in_class.append(class_node)
            functions_in_classes[node.name] = functions_in_class
    return functions_in_classes


def test_only_exception_files_modified():
    # print(f"Current working directory: {os.getcwd()}")  # Debugging line

    EXCEPTION_FILES = [
        "multigrid/envs/competative_red_blue_door.py",
        "multigrid/envs/__init__.py",
        "multigrid/rllib/models.py",
        "multigrid/rllib/ctde_torch_policy.py",
        "multigrid/scripts/train.py",
        "multigrid/scripts/visualize.py",
        "multigrid/scripts/train_ppo_cleanrl.py",
        "multigrid/scripts/train_sac_cleanrl.py",
        "multigrid/utils/training_utilis.py",
        "multigrid/wrappers.py",
        "multigrid/rllib/__init__.py"
    ]

    EXCEPTION_FOLDERS = ["submission/**", "notebooks/**"]

    for folder in EXCEPTION_FOLDERS:
        globbed_files = glob.glob(folder, recursive=True)
        # print(f"Adding files from folder {folder}: {globbed_files}")  # Debugging line
        EXCEPTION_FILES.extend(globbed_files)

    EXCEPTION_FILES = set(EXCEPTION_FILES)  # Converting to set for faster look-up

    # Get list of all files in the repository.
    all_files_result = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    all_files_result.check_returncode()

    all_files = set(all_files_result.stdout.splitlines())

    # Remove exception files from all_files to create the locked_files set.
    locked_files = all_files - EXCEPTION_FILES


    # Check if the tag 'v1.1' exists
    list_tags_result = subprocess.run(["git", "tag"], capture_output=True, text=True)
    tags = list_tags_result.stdout.splitlines()
    
    if "v1.1" in tags:
        base_commit = "v1.1"
    else:
        # If the tag doesn't exist, means it is in Github Classroom find the oldest (initial) commit hash
        oldest_commit_result = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"], 
            capture_output=True, text=True
        )
        base_commit = oldest_commit_result.stdout.strip()

    # Get list of changed files between HEAD and base_commit.
    changed_files_result = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"], 
        capture_output=True, text=True
    )
    # If the git command fails, the test should fail.
    changed_files_result.check_returncode()

    changed_files = set(changed_files_result.stdout.splitlines())

    # Check if any locked file is in the changed files list.
    modified_locked_files = changed_files & locked_files
    assert not modified_locked_files, f"Locked files were modified: {', '.join(modified_locked_files)}"


def test_restrict_file_changes():
    # Get the AST for the old version of the file
    old_code = subprocess.run(
        ["git", "show", "HEAD~1:multigrid/envs/competative_red_blue_door.py"], capture_output=True, text=True
    ).stdout
    old_tree = ast.parse(old_code)
    old_functions_in_classes = get_functions_in_classes_from_ast(old_tree)

    # Get the AST for the new version of the file
    with open("multigrid/envs/competative_red_blue_door.py", "r") as f:
        new_code = f.read()
    new_tree = ast.parse(new_code)
    new_functions_in_classes = get_functions_in_classes_from_ast(new_tree)

    for class_name, old_functions in old_functions_in_classes.items():
        new_functions = new_functions_in_classes.get(class_name, [])
        allowed_functions = ALLOWED_FUNCTIONS_IN_CLASS.get(class_name, [])
        for old_function, new_function in zip(old_functions, new_functions):
            if old_function.name not in allowed_functions:
                assert ast.dump(old_function) == ast.dump(
                    new_function
                ), f"Unauthorized modification in {class_name}.{old_function.name}"


