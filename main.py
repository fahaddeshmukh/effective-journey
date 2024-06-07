import os
import subprocess
import argparse

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), 'bin', script_name)
    print(f"Running {script_path}...")
    subprocess.run(["python", script_path], check=True)

def main(start_step):
    # Define the scripts in the order they need to be executed
    scripts = [
        "data_util.py",
        "word2vec.py",
        "sentence_transformer.py",
        "results.py"
    ]

    start_index = start_step - 1
    for script in scripts[start_index:]:
        run_script(script)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scripts in sequence with step selection.")
    parser.add_argument(
        "start_step",
        type=int,
        choices=[1, 2, 3, 4],
        help="Step number to start from: 1 for data_util.py, 2 for word2vec.py, 3 for sentence_transformer.py, 4 for results.py"
    )
    args = parser.parse_args()
    main(args.start_step)
