# Dataset should load data from the data json file.
# this should load up the dataset from a json file, split the dataset into train and tests, then load video into tensor frames, and return a list of dictionaries with input_ids, attention_mask, pixel_values, and labels
import json
import yaml
with open("../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)


dataset_path = params["dataset_path"]

class DatasetLoader():

    def __init__(self, verbose=True, dataset_file="wlasl_cleaned.json"):
        self.verbose = verbose
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        print(f"Loaded dataset with {len(self.dataset)} samples from {dataset_path}")
        pass





if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    print(dataset_path)
