import torch
from fusion import MultimodalFusion
from video import ClipFrameEncoder
from transcription import Transcribe
from VectorDB import VectorDBManager
import pickle
import sys

url = "https://www.youtube.com/watch?v=a3jHqWvgc1g"

transcribe = Transcribe(url)
transcribe.download()
transcribe.text()
transcribe.encode()
text_embs = transcribe.encoded_chunks

clipper = ClipFrameEncoder(url)
clipper.download()
clipper.encode()
video_embs = clipper.frame_embeddings

fuser = MultimodalFusion(text_embs, video_embs)
concat_embeddings = fuser.concat_fusion()

with open("embeddings_cache.pkl", "wb") as f:
    pickle.dump(concat_embeddings, f)

db = VectorDBManager()
db.store_embeddings(concat_embeddings, url)

print(f"Successfully stored embeddings for {url}")