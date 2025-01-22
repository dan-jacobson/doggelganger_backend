import asyncio
import aiohttp
from queue import Queue
from threading import Thread
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time

class AsyncImageProcessor:
    def __init__(self, model, batch_size=32, queue_size=1000, num_fetchers=8):
        self.model = model
        self.batch_size = batch_size
        self.image_queue = asyncio.Queue(maxsize=queue_size)
        self.processed_queue = Queue()
        self.num_fetchers = num_fetchers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
    async def fetch_image(self, session, url):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.read()
                    return data
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    async def producer(self, urls):
        async with aiohttp.ClientSession() as session:
            # Create tasks for all URLs but process them in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(urls), chunk_size):
                chunk_urls = urls[i:i + chunk_size]
                tasks = [self.fetch_image(session, url) for url in chunk_urls]
                
                # As images complete, add them to the queue immediately
                for task in asyncio.as_completed(tasks):
                    image_data = await task
                    if image_data:
                        await self.image_queue.put(image_data)
        
        # Signal that we're done producing
        await self.image_queue.put(None)

    def process_batch(self, batch):
        with torch.no_grad():
            embeddings = self.model(batch)
        return embeddings

    async def consumer(self):
        current_batch = []
        
        while True:
            # Get an image from the queue
            image_data = await self.image_queue.get()
            
            # Check for termination signal
            if image_data is None:
                if current_batch:  # Process any remaining images
                    batch_tensor = torch.stack(current_batch)
                    embeddings = self.process_batch(batch_tensor)
                    self.processed_queue.put(embeddings)
                break
                
            # Add to current batch
            current_batch.append(self.transform(image_data))
            
            # If we have a full batch, process it
            if len(current_batch) >= self.batch_size:
                batch_tensor = torch.stack(current_batch)
                embeddings = self.process_batch(batch_tensor)
                self.processed_queue.put(embeddings)
                current_batch = []

    async def run(self, urls):
        # Start producer and consumer
        producer_task = asyncio.create_task(self.producer(urls))
        consumer_task = asyncio.create_task(self.consumer())
        
        # Wait for both to complete
        await asyncio.gather(producer_task, consumer_task)
        
        # Signal that processing is complete
        self.processed_queue.put(None)

# Example usage
async def main():
    model = torch.jit.script(your_model)  # Your scripted model
    processor = AsyncImageProcessor(model)
    
    urls = [...]  # Your list of URLs
    
    await processor.run(urls)
    
    # Collect results
    results = []
    while True:
        embeddings = processor.processed_queue.get()
        if embeddings is None:
            break
        results.append(embeddings)

if __name__ == "__main__":
    asyncio.run(main())