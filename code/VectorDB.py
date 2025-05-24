import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
import hashlib

class VectorDBManager:
    def __init__(self, index_name: str = "multimodal-search"):
        self.dimension = 896
        self.index_name = index_name
        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION", "us-west-2"))
            )
        self.index = self.pc.Index(self.index_name)
    
    def store_embeddings(self, embeddings: List[Dict], youtube_url: str):
        url_hash = self._hash_url(youtube_url)
        vectors = []
        
        for item in embeddings:
            vectors.append({
                "id": self._generate_id(item, url_hash),
                "values": item["fused_embedding"].tolist(),
                "metadata": {
                    "text": str(item["text"]),
                    "start_time": float(item["start_time"]),
                    "end_time": float(item["end_time"]),
                    "timestamp": str(item["timestamp"]),
                    "duration": float(item["duration"]),
                    "fusion_method": str(item.get("fusion_method", "concat")),
                    "youtube_url": youtube_url,
                    "url_hash": url_hash
                }
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size])
    
    def _generate_id(self, item: Dict, url_hash: str) -> str:
        return f"{url_hash}_{item['start_time']:.0f}"
    
    def _hash_url(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def search(self, query_embedding: np.ndarray, youtube_url: str, top_k: int = 5) -> List[Dict]:
        url_hash = self._hash_url(youtube_url)
        
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"url_hash": {"$eq": url_hash}}
        )
        
        return [{
            "text": match.metadata.get("text", ""),
            "timestamp": match.metadata.get("timestamp", ""),
            "start_time": match.metadata.get("start_time", 0),
            "end_time": match.metadata.get("end_time", 0),
            "score": match.score,
            "youtube_url": match.metadata.get("youtube_url", "")
        } for match in results.matches]
    
    def search_all_videos(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [{
            "text": match.metadata.get("text", ""),
            "timestamp": match.metadata.get("timestamp", ""),
            "start_time": match.metadata.get("start_time", 0),
            "end_time": match.metadata.get("end_time", 0),
            "score": match.score,
            "youtube_url": match.metadata.get("youtube_url", "")
        } for match in results.matches]