from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            trust_remote_code=True,
                                            torch_dtype=torch.float16 if device != "cpu" else torch.float32, 
                                            device_map=None)

processor = AutoProcessor.from_pretrained(model_name)
model.to(device)
print("Model loaded successfully!")