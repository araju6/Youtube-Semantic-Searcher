import torch
from VectorDB import VectorDBManager
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import sys

    
url = "https://www.youtube.com/watch?v=tCPNBvwlnfI"

query = "defense"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model = clip_model.to(device)
clip_tokenizer = clip_processor.tokenizer

db = VectorDBManager()

with torch.no_grad():
    clip_inputs = clip_tokenizer(query, return_tensors='pt', padding=True, truncation=True).to(clip_model.device)
    clip_text_embedding = clip_model.get_text_features(**clip_inputs)
    clip_text_embedding = torch.nn.functional.normalize(clip_text_embedding, p=2, dim=1)
    clip_text_embedding = clip_text_embedding.cpu().numpy().flatten()

query_embedding = np.concatenate([clip_text_embedding, clip_text_embedding])
results = db.search(query_embedding, url, top_k=5)

print(f"Search results for '{query}':")
for r in results:
    print(f"Timestamp: {r['timestamp']}, Score: {r.get('score', 'N/A')}")