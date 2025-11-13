import os
import yaml
import time
from dotenv import load_dotenv
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
# Load environment variables from .env file
load_dotenv()
from pathlib import Path

CONFIG = None

# The 'with' statement is the best way to handle opening/closing files
with open('params/vlm.yml', 'r') as file:
    # Use yaml.safe_load() to parse the file
    # This converts the YAML into a Python dictionary (or list)
    try:
        CONFIG = yaml.safe_load(file)

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


# print(CONFIG["model"])
HF_TOKEN = os.getenv('HF_TOKEN')

class VLMModelLoader():
    def __init__(self):
        print("\n\n📍 Initializing VLMModelLoader...")
        self.model_name = CONFIG["model"]
        self.model=None
        self.processor=None
        self.prompt = CONFIG["prompt"]
        self.device = "cpu"
        self.load_model()

    
    def load_model(self):
        print(f"\n\n🚀 Starting {self.model_name} VLM Model Loading...")
        
        try:
            start_time = time.time()
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16"
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                device_map =self.device,)

            self.processor = AutoProcessor.from_pretrained(self.model_name,trust_remote_code=True)
            end_time = time.time()
            elapsed_time = end_time - start_time


            # self.model.to(self.device)

            print(f"\n\n✅ {self.model_name} VLM Model Loaded Successfully in ⏱️: {elapsed_time:.2f} seconds.")

        except Exception as e:
            print(f"Error loading model: {e}")  

        pass

    def generate_response(self):
        pass

    pass


if __name__ == "__main__":
    # vlm_loader = VLMModelLoader()
    print(Path.cwd())