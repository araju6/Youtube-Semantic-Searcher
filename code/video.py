import os
import torch
import yt_dlp
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class ClipFrameEncoder:
    def __init__(self, url, chunk_duration=5, frames_per_second=1):
        self.url = url
        self.chunk_duration = chunk_duration
        self.frames_per_second = frames_per_second
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.video_filepath = None
        self.frame_embeddings = []

    def download(self):
        download_folder = "temp_video"
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': os.path.join(download_folder, 'video_file.%(ext)s'),
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])
            self.video_filepath = os.path.join(download_folder, "video_file.mp4")
            print(f"Video downloaded to: {self.video_filepath}")
            return True
        except Exception as e:
            print(f"Error downloading video: {e}")
            return False

    def encode(self):
        if not self.video_filepath:
            print("No video file found.")
            return None
        
        cap = cv2.VideoCapture(self.video_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        frame_embeddings = []
        
        for chunk_start in np.arange(0, duration, self.chunk_duration):
            chunk_end = min(chunk_start + self.chunk_duration, duration)
            actual_chunk_duration = chunk_end - chunk_start
            
            total_frames_to_sample = int(actual_chunk_duration * self.frames_per_second)
            
            if total_frames_to_sample > 0:
                start_frame = int(chunk_start * fps)
                end_frame = int(chunk_end * fps)
                
                frame_indices = np.linspace(start_frame, end_frame - 1, 
                                          total_frames_to_sample, dtype=int)
                
                chunk_embeddings = []
                
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = cap.read()
                    
                    if success:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model.get_image_features(**inputs)
                            embedding = torch.nn.functional.normalize(outputs, p=2, dim=1)
                            chunk_embeddings.append(embedding.cpu().numpy().flatten())
                
                if chunk_embeddings:
                    mean_embedding = np.mean(chunk_embeddings, axis=0)
                    frame_embeddings.append(mean_embedding)
        
        cap.release()
        self.frame_embeddings = frame_embeddings
        return frame_embeddings

if __name__ == "__main__":
    clipper = ClipFrameEncoder("https://youtu.be/qHR1bGszwTU?si=GDDT9_s5ZRv6ji7x")
    clipper.download()
    clipper.encode()
    print(len(clipper.frame_embeddings), "embeddings generated (1 per 5-second chunk).")