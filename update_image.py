# Update docker and beaker image for the llm-server
import json
import argparse
import os
import time
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Start LLM server on Beaker interactive session.")
    parser.add_argument("--username", type=str, help="beaker username", default="harsh-trivedi")
    parser.add_argument("--force", action="store_true", help="delete the image if it already exists.")
    args = parser.parse_args()

    image_name = "llm-server"
    command = f"docker build -t {image_name} ."
    print(f"Running: {command}")
    subprocess.run(command, shell=True, stdout=open(os.devnull, 'wb'))

    command = f"beaker image inspect --format json {args.username}/{image_name}"
    try:
        image_is_present = subprocess.call(command, shell=True, stdout=open(os.devnull, 'wb')) == 0
    except:
        image_is_present = False

    if image_is_present:
        command = f"beaker image delete {args.username}/{image_name}"
        print(f"Running: {command}")
        subprocess.run(command, stdout=open(os.devnull, 'wb'), shell=True)

    command = f"beaker image create {image_name} --name {image_name} --workspace ai2/GPT3_Exps"
    print(f"Running: {command}")
    subprocess.run(command, shell=True, stdout=open(os.devnull, 'wb'))


if __name__ == "__main__":
    main()
