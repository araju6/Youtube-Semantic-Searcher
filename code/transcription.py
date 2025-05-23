import torch
import openai
import whisper
from transformers import AutoTokenizer, AutoModel
from pytube import YouTube
import os
import librosa
import numpy as np

class Transcribe:

    def __init__(self, url) -> None:
        self.url = url
        self.whisper = whisper.load_model("base")
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.encoder_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        self.audio_file = None
        self.transcription_chunks = []
        self.encoded_chunks = []

    def download(self):
        download_folder = "temp_audio"
        try:
            if not os.path.exists(download_folder):
                os.makedirs(download_folder)
            yt = YouTube(self.url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            if audio_stream:
                print("Downloading audio...")
                self.audio_filepath = audio_stream.download(output_path=download_folder, filename="audio_file.mp4")
                print(f"Audio downloaded to: {self.audio_filepath}")
                return True
            else:
                print("No audio stream found.")
                return False
        except Exception as e:
            print(f"Error downloading audio: {e}")
            return False
        
    def format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def text(self):
        self.download()
        if not self.audio_filepath:
            return None
        try:
            audio, sr = librosa.load(self.audio_filepath, sr=16000) 
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
