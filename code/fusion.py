import numpy as np
from sklearn.preprocessing import normalize

class MultimodalFusion:
    def __init__(self, text_chunks, video_embeddings):
        self.metadata = text_chunks
        self.text_emb = normalize(np.array([x['embedding'] for x in text_chunks]), axis=1)
        self.video_emb = normalize(np.array(video_embeddings), axis=1)
        
        if len(self.text_emb) != len(self.video_emb):
            raise ValueError(f"Mismatched counts: {len(self.text_emb)} text vs {len(self.video_emb)} video")

    def _create_output(self, fused_embeddings, method_name):
        return [{
            **self.metadata[i],
            'fused_embedding': fused_embeddings[i],
            'fusion_method': method_name,
            'video_embedding': self.video_emb[i]
        } for i in range(len(fused_embeddings))]

    def concat_fusion(self):
        fused = np.concatenate([self.text_emb, self.video_emb], axis=1)
        return self._create_output(fused, 'concat')

    def mean_fusion(self):
        text_padded = np.pad(self.text_emb, [(0,0), (0,128)], mode='constant')
        fused = (text_padded + self.video_emb) / 2
        return self._create_output(fused, 'mean')

    def weighted_mean_fusion(self, text_weight=0.6):
        text_padded = np.pad(self.text_emb, [(0,0), (0,128)], mode='constant')
        fused = text_weight*text_padded + (1-text_weight)*self.video_emb
        return self._create_output(fused, f'weighted_mean_{text_weight}')
