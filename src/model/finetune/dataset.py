# Dataset should load data from the data json file.
# this should load up the dataset from a json file, split the dataset into train and tests, then load video into tensor frames, and return a list of dictionaries with input_ids, attention_mask, pixel_values, and labels
import json
import yaml
from collections import defaultdict
import random
with open("../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)


dataset_path = params["dataset_path"]

class DatasetLoader():
    """
    Dataset loader for sign language video data with stratified splitting by gloss.
    
    This class loads processed WLASL dataset from JSON, performs stratified train/test split
    to ensure each sign language word (gloss) is proportionally represented in both sets,
    and prepares data for model training.
    """

    def __init__(self, verbose=True, dataset_file="wlasl_cleaned.json"):
        """
        Initialize the DatasetLoader with sign language video dataset.
        
        Args:
            verbose (bool, optional): Enable verbose logging. Defaults to True.
            dataset_file (str, optional): Name of the dataset JSON file. 
                                        Defaults to "wlasl_cleaned.json".
                                        
        Attributes:
            verbose (bool): Logging flag
            dataset (list): Complete dataset loaded from JSON
            train_data (list): Training split (set after calling _train_test_split)
            test_data (list): Test split (set after calling _train_test_split)
        """
        self.verbose = verbose
        
        print(f"\n\n 🔄 Starting Loading dataset procedure from {dataset_path}...")
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)

        print(f"Loaded dataset with {len(self.dataset)} samples from {dataset_path}")
        
        if self.verbose:
            print(f"\n\n 🔄 Starting Train Test Split...")
        # Initialize splits as None
        self.train_data = None
        self.test_data = None

        self._train_test_split(split_ratio=0.8)
        

        pass

    def _train_test_split(self, split_ratio=0.8):
        """
        Perform stratified train-test split ensuring each gloss is proportionally represented.
        
        This method groups samples by gloss (sign language word) and splits each group
        according to the specified ratio, ensuring balanced representation of all
        sign language words in both training and test sets.
        
        Args:
            split_ratio (float, optional): Proportion of data for training. Defaults to 0.8.
                                         
        Side Effects:
            Sets self.train_data and self.test_data with stratified splits
            
        Returns:
            tuple: (train_data, test_data) - Lists of training and test samples
        """
        
        
        # Group samples by gloss
        gloss_groups = defaultdict(list)
        for sample in self.dataset:
            gloss_groups[sample['gloss']].append(sample)
        
        train_data = []
        test_data = []
        
        # Split each gloss group proportionally
        for gloss, samples in gloss_groups.items():
            # Shuffle samples for random split
            random.shuffle(samples)
            
            # Calculate split index
            split_idx = int(len(samples) * split_ratio)
            
            # Ensure at least 1 sample in train if possible
            if len(samples) > 1 and split_idx == 0:
                split_idx = 1
            # Ensure at least 1 sample in test if possible
            elif len(samples) > 1 and split_idx == len(samples):
                split_idx = len(samples) - 1
            
            # Split the samples
            train_samples = samples[:split_idx]
            test_samples = samples[split_idx:]
            
            train_data.extend(train_samples)
            test_data.extend(test_samples)
            
            if self.verbose:
                print(f"Gloss '{gloss}': {len(train_samples)} train, {len(test_samples)} test samples")
        
        # Shuffle the final datasets
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        self.train_data = train_data
        self.test_data = test_data
        
        if self.verbose:
            print(f"\nTotal: {len(train_data)} train, {len(test_data)} test samples")
            print(f"Split ratio: {len(train_data)/(len(train_data)+len(test_data)):.2%} train")
        
        return train_data, test_data

    def _parse_video_frames(self, video_path):
        """
        Extract and process frames from video file for model input.
        
        Args:
            video_path (str): Path to the video file relative to the dataset location
            
        Returns:
            torch.Tensor: Processed video frames ready for model input
            
        Note:
            This method needs implementation based on your video processing requirements
            and the specific frame extraction strategy for your VLM model.
        """
        pass

    def collate_fn(self, batch):
        """
        Collate function for DataLoader to batch samples together.
        
        Args:
            batch (list): List of processed samples from the dataset
            
        Returns:
            dict: Batched data with keys:
                - input_ids: Tokenized text inputs
                - attention_mask: Attention masks for text
                - pixel_values: Processed video frames
                - labels: Target labels for training
                
        Note:
            This method needs implementation based on your model's input requirements.
        """
        pass

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        
        Returns:
            int: Total number of samples in the loaded dataset
        """
        return len(self.dataset)
    

    def preprocess_data(self, data_point):
        """
        Preprocess a single data point for model training.
        
        Args:
            data_point (dict): Single sample with keys:
                - prompt: Instruction text for the model
                - video_path: Path to the video file
                - gloss: Target sign language word
                
        Returns:
            dict: Preprocessed sample with model-ready tensors:
                - input_ids: Tokenized prompt
                - attention_mask: Attention mask for tokens
                - pixel_values: Processed video frames
                - labels: Target labels for training
                
        Note:
            This method needs implementation based on your VLM model's preprocessing pipeline.
        """
        pass






if __name__ == "__main__":
    dataset_loader = DatasetLoader()
    print(dataset_path)
