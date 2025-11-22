# Dataset should load data from the data json file.
# this should load up the dataset from a json file, split the video into tensor frames, and return a list of dictionaries with input_ids, attention_mask, pixel_values, and labels


class DatasetLoader():

    def __init__(self, verbose=True, dataset_file="wlasl_cleaned.json"):
        self.verbose = verbose
        self.dataset_file = dataset_file
        self.dataset = []
        self.load_dataset()
        pass