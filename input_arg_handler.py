# TODO: Programmer info
# Name: Fahad Moidy
# Date: 5/03/2022

# File Info
# Last updated: 03/05/2022
# Description: Handler used to process the in line arguments for the parent program

# Python imports
from sys import argv, exit
from os import path

# Fetch input args storing them as a key/value pair for the parent program to use
def train_fetch_input_args():
    # Check if any args were parsed
    if len(argv) > 1:
        # Command line args parsed
        # Validate data dir exists
        if not path.isdir(argv[1]):
            print("Directory: %s is invalid!" % (argv[1]))
            exit(1)
        
        # Create dictionary to store program args
        args_dict = {
            "data_dir" : str(argv[1]),
            "gpu" : bool(0),
            "save_dir" : "./",
            "arch" : "vgg16",
            "learning_rate" : float(0.001),
            "hidden_units" : int(5000),
            "epochs" : int(15)
        }

        counter = 0
        for arg in argv:
            counter += 1
            if arg == '--save_dir':
                args_dict["save_dir"] = argv[counter]
                # Validate dir exists
                if not path.isdir(args_dict["save_dir"]):
                    print("Directory: %s is invalid!" % (args_dict["save_dir"]))
                    exit(1)

            if arg == '--arch':
                args_dict["arch"] = argv[counter]

            if arg == '--learning_rate':
                args_dict["learning_rate"] = float(argv[counter])

            if arg == '--hidden_units':
                args_dict["hidden_units"] = int(argv[counter])

            if arg == '--epochs':
                args_dict["epochs"] = int(argv[counter])

            if arg == '--gpu':
                args_dict["gpu"] = bool(1)

        return args_dict

    else:
        # No arguments parsed
        print("Usage: python %s data_directory" % (argv[0]))
        exit(1)
        
# Fetch input args storing them as a key/value pair for the parent program to use
def predict_fetch_input_args():
    # Check if any args were parsed
    if len(argv) > 2:
        # Command line args parsed
        # Validate image exists
        if not path.isfile(argv[1]):
            print("Image: %s is invalid!" % (argv[1]))
            exit(1)
        
        # Validate checkpoint exists
        if not path.isfile(argv[2]):
            print("Checkpoint: %s is invalid!" % (argv[2]))
            exit(1)
        
        # Create dictionary to store program args
        args_dict = {
            "image" : str(argv[1]),
            "checkpoint" : str(argv[2]),
            "gpu" : bool(0),
            "top_k" : int(5),
            "category_names" : str("cat_to_name.json")
        }

        counter = 0
        for arg in argv:
            counter += 1

            if arg == '--top_k':
                args_dict["top_k"] = int(argv[counter])

            if arg == '--gpu':
                args_dict["gpu"] = bool(1)

        return args_dict

    else:
        # No arguments parsed
        print("Usage: python %s /path/to/image checkpoint" % (argv[0]))
        exit(1)
