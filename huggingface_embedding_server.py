import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import io


nest_asyncio.apply()

# Get ngrok authtoken from environment if available
NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN')
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create FastAPI app
app = FastAPI(
    title="Hugging Face LlamaIndex Embedding API",
    description="Embed text, documents, images, or multimodal using HuggingFace",
    version="1.1.0"
)

# Load text embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load CLIP model for image & multimodal
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Schemas
class SingleTextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

# Endpoints
@app.get("/")
async def root():
    return {"message": "Embedding API is running!"}

@app.post("/embed_text", response_model=EmbeddingResponse)
async def embed_single_text(request: SingleTextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")
    embedding = embed_model.get_text_embedding(request.text)
    return EmbeddingResponse(embeddings=[embedding])

@app.post("/embed_docs", response_model=EmbeddingResponse)
async def embed_documents(request: BatchTextRequest):
    texts = request.texts
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    if len(texts) == 1 and isinstance(texts[0], list):
        texts = texts[0]
    if not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=400, detail="Each item in 'texts' must be a string.")
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="No valid texts provided after filtering empty strings")
    try:
        if len(texts) == 1:
            embedding = embed_model.get_text_embedding(texts[0])
            embeddings = [embedding]
        else:
            embeddings = [embed_model.get_text_embedding(text) for text in texts]
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {str(e)}")

@app.post("/embed_batch", response_model=EmbeddingResponse)
async def embed_batch_file(file: UploadFile = File(...)):
    contents = await file.read()
    text_data = contents.decode("utf-8").splitlines()
    text_data = [line.strip() for line in text_data if line.strip()]
    if not text_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty or contains no valid text.")
    try:
        embeddings = [embed_model.get_text_embedding(text) for text in text_data]
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing embeddings: {str(e)}")

@app.post("/embed_image", response_model=EmbeddingResponse)
async def embed_image_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        image_emb = clip_model.get_image_features(**inputs)
        image_emb = image_emb[0].detach().tolist()
        return EmbeddingResponse(embeddings=[image_emb])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/embed_multimodal", response_model=EmbeddingResponse)
async def embed_multimodal(text: str, file: UploadFile = File(...)):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No text provided")
    try:
        text_inputs = clip_processor(text=text, return_tensors="pt", padding=True)
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb = text_emb[0].detach().tolist()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_inputs = clip_processor(images=image, return_tensors="pt")
        image_emb = clip_model.get_image_features(**image_inputs)
        image_emb = image_emb[0].detach().tolist()
        return EmbeddingResponse(embeddings=[text_emb, image_emb])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing multimodal input: {str(e)}")


if NGROK_AUTH_TOKEN and os.getenv("RUN_WITH_NGROK", "1") == "1":
    ngrok_tunnel = ngrok.connect(8000)
    print('Public URL:', ngrok_tunnel.public_url)

if __name__ == "__main__":
    uvicorn.run("huggingface_embedding_server:app", host="0.0.0.0", port=8000, reload=True) 