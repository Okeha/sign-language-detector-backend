# from model_loader import VLMModelLoader
import time
from transformers import Trainer, TrainingArguments, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from datasets import Dataset
from dataset import DatasetLoader
import os
import json
import yaml
import torch
import torch_directml
from dotenv import load_dotenv
load_dotenv()
import decord
from peft import LoraConfig, get_peft_model

HF_TOKEN = os.getenv('HF_TOKEN')
NUM_FRAMES = 4

with open("../params/vlm.yml", "r") as f:
    params = yaml.safe_load(f)

class VLMFineTuneTrainer():
    def __init__(self, verbose=True):
        self.verbose = verbose
        # self.vlm_loader = VLMModelLoader()
        self.model_name = params["model"]
        self.model = None
        self.processor = None
        self.device = None
        self.device_type = None
        
        # Set device before loading model
        self._set_device()
        
        # Load model
        self.load_model()
        
        # Set LoRA config and wrap model
        self.peft_model = get_peft_model(self.model, self._set_lora_config())
        self.train()
        pass

    def _set_device(self):
        # --- NEW HARDWARE DETECTION LOGIC ---
        if torch.cuda.is_available():
            self.device_type = "cuda"
            self.device = torch.device("cuda")
            print(f"✅ Found NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        elif torch_directml.is_available():
            self.device_type = "dml"
            print(torch_directml.device())
            self.device = torch_directml.device() # Get the DML device
            
        else:
            self.device_type = "cpu"
            self.device = torch.device("cpu")
            print("❌ No GPU found. Falling back to CPU.")
    
    def load_model(self):
        if self.verbose:
            print(f"\n\n🚀 Starting {self.model_name} VLM Model Loading...")
        
        try:
            
            start_time = time.time()
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16"
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name,trust_remote_code=True, token=HF_TOKEN)

            if self.device_type == "dml":
                if self.verbose:
                    print("--- Loading for AMD (DirectML) in float16 ---")
                    print("⚠️  4-bit quantization is NOT supported on DML")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,                    
                    dtype=torch.float16,    # Load in float16
                    token=HF_TOKEN,
                    low_cpu_mem_usage=True,
                
                ).to(self.device)
                
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    quantization_config=bnb_config,
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN,
                    device_map =self.device,
                    )
            end_time = time.time()
            elapsed_time = end_time - start_time


            # self.model.to(self.device)

            print(f"\n\n✅ {self.model_name} VLM Model Loaded Successfully in ⏱️: {elapsed_time:.2f} seconds.")

        except Exception as e:
            print(f"Error loading model: {e}")  

        pass

    # ---------------------------------------------------------
    # FRAME SAMPLING
    # ---------------------------------------------------------
    def sample_frames(self, video_path: str, num_frames: int = NUM_FRAMES):
        try:
            video_path = f"data_engineering/{video_path}"
            vr = decord.VideoReader(video_path)
            total = len(vr)
            indices = torch.linspace(0, total - 1, num_frames).long()
            frames = vr.get_batch(indices).asnumpy()   # (T, H, W, C)
            return frames
        except Exception as e:
            print(f"Error sampling frames from {video_path}: {e}")
            return None

    # ---------------------------------------------------------
    # COLLATOR
    # ---------------------------------------------------------
    def video_collate(self, batch):
        videos = [self.sample_frames(item["video_path"], NUM_FRAMES) for item in batch]
        prompts = [item["prompt"] for item in batch]
        answers = [item["gloss"] for item in batch]

        # MUST match HF requirements: <video> placeholder!
        texts = [f"<video>USER: {p}\nASSISTANT: {a}" for p, a in zip(prompts, answers)]

        model_inputs = self.processor(
            videos=videos,
            text=texts,
            padding=True,
            return_tensors="pt",
        )

        # causal LM labels = clone input_ids
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

    # ---------------------------------------------------------
    # LORA CONFIG
    # ---------------------------------------------------------
    def _set_lora_config(self):
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )

    def train(self):
        dataset_loader = DatasetLoader(verbose=self.verbose)
        train_dataset = dataset_loader.train_data
        test_dataset = dataset_loader.test_data

        # # Preprocess datasets
        # train_data = [self.preprocess_data(sample["video_path"]) for sample in train_dataset]
        # test_data = [self.preprocess_data(sample["video_path"]) for sample in test_dataset]

        
        hf_train_dataset = Dataset.from_list(train_dataset)
        hf_test_dataset = Dataset.from_list(test_dataset)

        training_args = TrainingArguments(
            output_dir="./vlm_finetuned",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=1,
            gradient_accumulation_steps=4,
            # fp16=True,
            fp16=(self.device_type == "cuda"),
            # bf16=False,
            logging_steps=10,
            save_strategy="epoch",
            # evaluation_strategy="epoch",
            remove_unused_columns=False,     # IMPORTANT
            # report_to="tensorboard",
            optim="adamw_torch",              # ← SIMPLE ADAM
        )

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_test_dataset,
            data_collator=self.video_collate,
        )

        start_time = time.time()
        if self.verbose:
            print("\n\n🧠 Starting Fine-Tuning...\n")

        trainer.train()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if self.verbose:
            print("\n\n✅ Fine-Tuning Complete! in ⏱️: {elapsed_time:.2f} seconds.")

        
        pass


if __name__ == "__main__":
    vlm_finetune_trainer = VLMFineTuneTrainer()