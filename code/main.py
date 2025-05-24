import torch
from fusion import MultimodalFusion
from video import ClipFrameEncoder
from transcription import Transcribe
from VectorDB import VectorDBManager
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle

url = "https://www.youtube.com/watch?v=KRqg3RJFWPo&t=224s"
query = "Green"
transcribe = Transcribe(url)

# transcribe.download()
# transcribe.text()
# transcribe.encode()
# text_embs = transcribe.encoded_chunks

# clipper = ClipFrameEncoder(url)
# clipper.download()
# clipper.encode()
# video_embs = clipper.frame_embeddings
# #fusion
# fuser = MultimodalFusion(text_embs, video_embs)
# concat_embeddings = fuser.concat_fusion()

# with open("embeddings_cache.pkl", "wb") as f:
#     pickle.dump(concat_embeddings, f)

# try:
#     with open("embeddings_cache.pkl", "rb") as f:
#         concat_embeddings = pickle.load(f)
#     print("Loaded embeddings from cache")
# except FileNotFoundError:
#     print("No cache found, processing from scratch")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to(transcribe.device)
clip_tokenizer = clip_processor.tokenizer

with open("embeddings_cache.pkl", "rb") as f:
        concat_embeddings = pickle.load(f)
        print("Loaded embeddings from cache")

db = VectorDBManager()
db.store_embeddings(concat_embeddings)

with torch.no_grad():
    bert_inputs = transcribe.tokenizer(query, return_tensors='pt', 
                                     padding=True, truncation=True, 
                                     max_length=512).to(transcribe.device)
    bert_output = transcribe.encoder_model(**bert_inputs)
    bert_embedding = transcribe.mean_pooling(bert_output, bert_inputs['attention_mask'])
    bert_embedding = torch.nn.functional.normalize(bert_embedding, p=2, dim=1)
    bert_embedding = bert_embedding.cpu().numpy().flatten()

with torch.no_grad():
    clip_inputs = clip_tokenizer(query, return_tensors='pt', 
                               padding=True, truncation=True).to(clip_model.device)
    clip_text_embedding = clip_model.get_text_features(**clip_inputs)
    clip_text_embedding = torch.nn.functional.normalize(clip_text_embedding, p=2, dim=1)
    clip_text_embedding = clip_text_embedding.cpu().numpy().flatten()  # 512D


query_embedding = np.concatenate([bert_embedding, clip_text_embedding])


results = db.search(query_embedding, top_k=5)
print(results)
for r in results:
    print(r['timestamp'])