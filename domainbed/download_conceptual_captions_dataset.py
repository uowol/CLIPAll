from datasets import load_dataset
# import pickle as pkl

# Get the dataset
image_data = load_dataset("conceptual_captions", name="labeled", split="train", 
                            cache_dir="/data4/kchanwo/.cache/huggingface/datasets")