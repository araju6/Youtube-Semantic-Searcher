import torch
import whisper
from transformers import AutoTokenizer, AutoModel
import os
import librosa
import numpy as np
import yt_dlp

class Transcribe:
    def __init__(self, url, batch_size=32) -> None:
        self.url = url
        self.batch_size = batch_size
        self.whisper = whisper.load_model("base")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_model = self.encoder_model.to(self.device)
        self.audio_filepath = None
        self.transcription_chunks = []
        self.encoded_chunks = []

    def download(self):
        download_folder = "temp_audio"
        try:
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(download_folder, 'audio_file.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'quiet': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            self.audio_filepath = os.path.join(download_folder, "audio_file.mp3")
            print(f"Audio downloaded to: {self.audio_filepath}")
            return True

        except Exception as e:
            print(f"Error downloading audio: {e}")
            return False
        
    def format_timestamp(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def text(self):
        if not self.audio_filepath:
            return None
        try:
            audio, sr = librosa.load(self.audio_filepath, sr=16000, dtype=np.float32)
            duration = len(audio) / sr
            chunk_duration = 5.0
            
            chunks = []
            print(f"Processing {duration:.1f} seconds of audio in {chunk_duration}s chunks...")
            for start_time in np.arange(0, duration, chunk_duration):
                end_time = min(start_time + chunk_duration, duration)
                
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_chunk = audio[start_sample:end_sample]
                
                if len(audio_chunk) > 0:
                    result = self.whisper.transcribe(audio_chunk)
                    chunk_data = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'text': result['text'].strip(),
                        'timestamp': self.format_timestamp(start_time),
                        'duration': end_time - start_time
                    }
                    
                    if chunk_data['text']:
                        chunks.append(chunk_data)
            self.transcription_chunks = chunks
            return chunks
        except Exception as e:
            print("Failed to transcribe")
            return None

    def encode(self):
        if not self.transcription_chunks:
            return None
        
        try:
            texts = [chunk['text'] for chunk in self.transcription_chunks if chunk['text'].strip()]
            batch_count = (len(texts) // self.batch_size) + (1 if len(texts) % self.batch_size else 0)
            all_embeddings = np.zeros((len(texts), 384), dtype=np.float32)
            
            for i in range(batch_count):
                batch_texts = texts[i*self.batch_size : (i+1)*self.batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
                
                with torch.no_grad():
                    outputs = self.encoder_model(**inputs)
                    embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    all_embeddings[i*self.batch_size : (i+1)*self.batch_size] = embeddings.cpu().numpy()
            
            chunk_idx = 0
            encoded_chunks = []
            for chunk in self.transcription_chunks:
                if chunk['text'].strip():
                    encoded_chunk = {
                        'start_time': chunk['start_time'],
                        'end_time': chunk['end_time'],
                        'text': chunk['text'],
                        'timestamp': chunk['timestamp'],
                        'embedding': all_embeddings[chunk_idx],
                        'duration': chunk['duration']
                    }
                    encoded_chunks.append(encoded_chunk)
                    chunk_idx += 1
            
            self.encoded_chunks = encoded_chunks
            return encoded_chunks
            
        except Exception as e:
            print(f"Failed to encode")
            return None
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

if __name__ == "__main__":
    temp = Transcribe("https://youtu.be/qHR1bGszwTU?si=GDDT9_s5ZRv6ji7x")
    temp.download()
    temp.audio_filepath = "temp_audio/audio_file.mp3"
    temp.text()
    temp.encode()
    print(temp.transcription_chunks)