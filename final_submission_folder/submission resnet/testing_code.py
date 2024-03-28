# You are free to either implement both test() and evaluate() function, or implement test_batch() and evaluate_batch() function. Apart from the 2 functions which you must mandatorily implement, you are free to implement some helper functions as per your convenience.

# Import all necessary python libraries here
# Do not write import statements anywhere else
import os
import pandas as pd
from training_code import CustomResNet18, AudioDataset as AD,transformations,BATCH_SIZE,device
import torch
from torch.utils.data import DataLoader

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "/home/pc/test_data"
OUTPUT_CSV_ABSOLUTE_PATH = "/home/pc/output.csv"

TEST_DATA_DIRECTORY_ABSOLUTE_PATH = "overfitting_val/car_horn"
OUTPUT_CSV_ABSOLUTE_PATH = "tempfolder/output.csv"

# The above two variables will be changed during testing. The current values are an example of what their contents would look like.

state_dict = state_dict = torch.load('epoch50.pth', map_location=torch.device(device))
model = CustomResNet18()
model.load_state_dict(state_dict)
# def evaluate(file_path):
#     # Write your code to predict class for a single audio file instance here
#     return predicted_class

def evaluate_batch(test_loader_batch):
    # Write your code to predict class for a batch of audio file instances here
    model.eval()
    outputs =  model(test_loader_batch).to(device)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted + torch.ones_like(predicted).to(device)

    # return predicted_class_batch
    return predicted.tolist()


# def test():
#     filenames = []
#     predictions = []
#     # for file_path in os.path.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
#     for file_name in os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH):
#         # prediction = evaluate(file_path)
#         absolute_file_name = os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, file_name)
#         prediction = evaluate(absolute_file_name)

#         filenames.append(absolute_file_name)
#         predictions.append(prediction)
#     pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


def test_batch(batch_size=1):
    filenames = []
    predictions = []

    filenames = os.listdir(TEST_DATA_DIRECTORY_ABSOLUTE_PATH)
    filenames = ([os.path.join(TEST_DATA_DIRECTORY_ABSOLUTE_PATH, i) for i in filenames])

    test_dataset = AD(TEST_DATA_DIRECTORY_ABSOLUTE_PATH,testing=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)
    with torch.no_grad(): 
        for inputs,_ in test_loader:
            inputs = inputs.to(device)
            predictions.extend(evaluate_batch(inputs))
    pd.DataFrame({"filename": filenames, "pred": predictions}).to_csv(OUTPUT_CSV_ABSOLUTE_PATH, index=False)


# Uncomment exactly one of the two lines below, i.e. either execute test() or test_batch()
# test()
test_batch()