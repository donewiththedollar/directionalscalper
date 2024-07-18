import os
from shutil import copy
import json
import argparse

IGNORE_CATEGORY = ["messengers","logger","hotkeys","exchanges","blacklist","whitelist"]
FILE_DIRECTORY: str = __file__
FILE_DIRECTORY = FILE_DIRECTORY.removesuffix(os.path.basename(FILE_DIRECTORY))
EXAMPLE_FILE = FILE_DIRECTORY+"config_example.json"

def get_config_file(arg) -> str:
    while True: 
        if isinstance(arg.config,str):
            i_config_path = arg.config
        else:
            i_config_path: str = input("Name the config file you want to update:")
        print(i_config_path)
        if i_config_path.startswith("."):
            i_config_path = input_with_dot(i_config_path)
        elif not i_config_path.startswith("/"):
            i_config_path = FILE_DIRECTORY+i_config_path
        if os.path.isfile(i_config_path):
            return i_config_path
        config_name = i_config_path.split("/")[-1]
        create_new_config(i_config_path, config_name)
        
def input_with_dot(config_path) -> str:
    """Makes the path for the config file with a dot declaration"""
    config_path = config_path.removeprefix("./")
    config_path = FILE_DIRECTORY + config_path
    return config_path

def create_new_config(config_path: str, config_input: str) -> None:
    print(f"Can't find a file at this path : {config_path}")
    want_create: str = input("Would you like to create one ? (y/n)")
    if want_create == "y" or want_create == "yes":
        try:
            truncated_config_path = config_path.removesuffix(config_input)
            os.makedirs(truncated_config_path,exist_ok=True)
            copy(EXAMPLE_FILE, config_path)
            exit()
        except PermissionError as e:
            print("I can't write a file here, please select somewhere else.")
            print(e)
            print("\n\n")
    if want_create == "n" or want_create == "no":
        return
    else:
        print("You need to answer with exactly yes|y or no|n. (One of the four)")
        return create_new_config(config_path, config_input)

def create_dict(source: dict, config = None) -> dict:
    destination = {
        key:check_value(key,value,config) if key in IGNORE_CATEGORY
        else create_dict(value, config) if isinstance(value,dict)
        
        else [
            create_dict(item, config) if isinstance(item,dict)
            else None
            for item in value
        ]
        if isinstance(value,list)
        
        else check_value(key, value, config) if config is not None
        else value
        for key, value in source.items()
    }
    
    return destination

def check_value(i_key, i_value, config) -> str | int | float | bool:
    stack = [config]
    while stack:
        d = stack.pop()
        if i_key in d:
            return d[i_key]
        for key, value in d.items():
            if isinstance(value, dict):
                stack.append(value)
    return i_value

def main():
    """
    This script asks for the config file you want to edit, and if it does not exists in the Path it finds, It will create a new instance of this file.
    If there was already an existing file, It will compare the old version with the new one, keeping the values of the old one while using only the keys of the new dictionary.
    """
    parser = argparse.ArgumentParser(
        usage = "write python3 update_config.py and give it your config file path, either an absolute path, or a relative path.",
        description= 
            """
            This script asks for the config file you want to edit, and if it does not exists in the Path it finds, It will create a new instance of this file.
            If there was already an existing file, It will only keep the still existing values of the old config file.
            """
        )
    parser.add_argument("--config", metavar="config", type=str, help="Enter your config path, either absolute or relative path accepted.")
    arg = parser.parse_args()
    
    
    config_path = get_config_file(arg)
    with open(config_path) as config_f, open(EXAMPLE_FILE) as example_f, open(config_path+".new", "w") as new:
        config, example = json.load(config_f), json.load(example_f)
        new_dict: dict = {}
        new_dict = create_dict(example, config)
        json.dump(new_dict, new, indent = 4)
    print("Your file has been successfuly updated.")
    print("""
    If the file already existed, look for the new filename.json.new.
    It's up to you to check if everything seems ok and to rename it to filename.json to replace the previous file.
    
    If the file did not exist previously, It should already be named filename.ext (ext being the extension you wrote).
          """)


if __name__ == "__main__":
    main()
