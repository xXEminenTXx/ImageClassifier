# Programmer info
# Name: Fahad Moidy
# Date: 5/03/2022

# Example command
# python predict.py flowers/test/1/image_06743.jpg ./saves/checkpoint.pth --gpu

# Python Imports
import json


# Program dependacies
from input_arg_handler import predict_fetch_input_args
from predict_functions import load_checkpoint, process_image, predict

# Define main function
def main():
    # Handle command line args (or lack of)
    args = predict_fetch_input_args()
    
    print("Processing image: %s, using checkpoint: %s. GPU enabled: %s. Top K: %d. Category Names File: %s" % (
        args["image"], args["checkpoint"], args["gpu"], args["top_k"], args["category_names"]))
    
    
    # function that loads a checkpoint and rebuilds the model
    model = load_checkpoint(args["checkpoint"])
    print(model)

    
    # Processes Image   
    image = process_image(args["image"])
    
    
    # Predict the class from an image file
    probs, classes = predict(args["image"], model, args["top_k"], args["gpu"])
    
    
    # Load category map
    with open(args["category_names"], 'r') as f:
        cat_to_name = json.load(f)
        
    flower_names = [cat_to_name[i] for i in classes]

    k = 0
    for i in classes:
        print("Name: %s, Probability: %f" % (flower_names[k] , probs[k]))
        k += 1
        
    print("Done!")
    
#   Call main function, starting the program
if __name__ == "__main__":
    main()  
