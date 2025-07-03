import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import io
import chromadb
from google.colab import userdata
from chromadb.config import Settings

nest_asyncio.apply()

# Get ngrok authtoken from environment if available
NGROK_AUTH_TOKEN = userdata.get("NGROK_AUTH_TOKEN")
if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "image_embeddings"

# Create FastAPI app
app = FastAPI(
    title="Hugging Face LlamaIndex Embedding API",
    description="Embed text, documents, images, or multimodal using HuggingFace with ChromaDB vector search",
    version="2.0.0",
)

# Load text embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Load CLIP model for image & multimodal
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


# Initialize ChromaDB
def initialize_chroma_db():
    """Initialize ChromaDB client and collection"""
    try:
        # Create ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # Get or create collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(
                f"Loaded existing collection '{COLLECTION_NAME}' with {collection.count()} items"
            )
        except:
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},  # Using cosine similarity
            )
            print(f"Created new collection '{COLLECTION_NAME}'")

        return client, collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        raise


# Initialize ChromaDB
chroma_client, chroma_collection = initialize_chroma_db()


# Vector Database Class for ChromaDB
class ChromaVectorDatabase:
    def __init__(self, collection):
        self.collection = collection

    def add_image(
        self, image_path: str, embedding: List[float], metadata: Dict[str, Any] = None
    ):
        """Add an image and its embedding to ChromaDB"""
        doc_id = str(uuid.uuid4())

        # Prepare metadata
        full_metadata = {
            "image_path": image_path,
            "created_at": datetime.now().isoformat(),
            "type": "image",
        }
        if metadata:
            full_metadata.update(metadata)

        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[image_path],  # Using image path as document
            metadatas=[full_metadata],
            ids=[doc_id],
        )

        return doc_id

    def search_by_text(
        self, text_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar images using text embedding"""
        try:
            results = self.collection.query(
                query_embeddings=[text_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["distances"][0],
                    )
                ):
                    formatted_results.append(
                        {
                            "similarity": 1
                            - distance,  # Convert distance to similarity
                            "image_path": metadata.get("image_path", doc),
                            "metadata": metadata,
                            "id": results["ids"][0][i]
                            if results["ids"]
                            else f"result_{i}",
                        }
                    )

            return formatted_results
        except Exception as e:
            print(f"Error searching ChromaDB: {e}")
            return []

    def get_all_images(self) -> List[Dict[str, Any]]:
        """Get all stored images"""
        try:
            # Get all items from collection
            results = self.collection.get(
                include=["documents", "metadatas", "embeddings"]
            )

            formatted_results = []
            if results["documents"]:
                for i, (doc, metadata) in enumerate(
                    zip(results["documents"], results["metadatas"])
                ):
                    formatted_results.append(
                        {
                            "id": results["ids"][i],
                            "image_path": metadata.get("image_path", doc),
                            "metadata": metadata,
                            "created_at": metadata.get("created_at", "unknown"),
                        }
                    )

            return formatted_results
        except Exception as e:
            print(f"Error getting all images: {e}")
            return []

    def image_exists(self, image_path: str) -> bool:
        """Check if an image already exists in the database"""
        try:
            results = self.collection.get(
                where={"image_path": image_path}, include=["documents"]
            )
            return len(results["documents"]) > 0
        except Exception as e:
            print(f"Error checking image existence: {e}")
            return False

    def get_count(self) -> int:
        """Get total number of items in the database"""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Error getting count: {e}")
            return 0

    def delete_image(self, image_id: str) -> bool:
        """Delete an image from the database"""
        try:
            self.collection.delete(ids=[image_id])
            return True
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False


# Initialize vector database
vector_db = ChromaVectorDatabase(chroma_collection)


# Schemas
class SingleTextRequest(BaseModel):
    text: str


class BatchTextRequest(BaseModel):
    texts: List[str]


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    similarity: float
    image_path: str
    metadata: Dict[str, Any]
    id: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int


class ProcessingResult(BaseModel):
    message: str
    processed_images: int
    skipped_images: int
    total_images: int


class DatabaseStats(BaseModel):
    total_images: int
    database_path: str
    uploads_directory: str
    collection_name: str
    database_type: str


def generate_image_embedding(image: Image.Image) -> List[float]:
    """Generate embedding for an image using CLIP model"""
    inputs = clip_processor(images=image, return_tensors="pt")
    image_emb = clip_model.get_image_features(**inputs)
    return image_emb[0].detach().tolist()


def generate_text_embedding_for_search(text: str) -> List[float]:
    """Generate embedding for text using CLIP model for image search"""
    text_inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    text_emb = clip_model.get_text_features(**text_inputs)
    return text_emb[0].detach().tolist()


# Existing Endpoints
@app.get("/")
async def root():
    return {"message": "Enhanced Embedding API with ChromaDB is running!"}


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
        raise HTTPException(
            status_code=400, detail="Each item in 'texts' must be a string."
        )
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise HTTPException(
            status_code=400,
            detail="No valid texts provided after filtering empty strings",
        )
    try:
        if len(texts) == 1:
            embedding = embed_model.get_text_embedding(texts[0])
            embeddings = [embedding]
        else:
            embeddings = [embed_model.get_text_embedding(text) for text in texts]
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing embeddings: {str(e)}"
        )


@app.post("/embed_batch", response_model=EmbeddingResponse)
async def embed_batch_file(file: UploadFile = File(...)):
    contents = await file.read()
    text_data = contents.decode("utf-8").splitlines()
    text_data = [line.strip() for line in text_data if line.strip()]
    if not text_data:
        raise HTTPException(
            status_code=400, detail="Uploaded file is empty or contains no valid text."
        )
    try:
        embeddings = [embed_model.get_text_embedding(text) for text in text_data]
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing embeddings: {str(e)}"
        )


@app.post("/embed_image", response_model=EmbeddingResponse)
async def embed_image_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Generate unique filename
        file_extension = Path(file.filename).suffix if file.filename else ".jpg"
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOADS_DIR / unique_filename

        # Save image to uploads folder
        image.save(file_path)

        # Generate embedding
        embedding = generate_image_embedding(image)

        # Store in ChromaDB
        metadata = {
            "original_filename": file.filename or "unknown",
            "file_size": len(contents),
            "image_format": image.format or "unknown",
            "image_size": f"{image.size[0]}x{image.size[1]}"
            if image.size
            else "unknown",
        }
        doc_id = vector_db.add_image(str(file_path), embedding, metadata)

        return EmbeddingResponse(embeddings=[embedding])
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
        raise HTTPException(
            status_code=500, detail=f"Error processing multimodal input: {str(e)}"
        )


# New Endpoints with ChromaDB
@app.post("/search_image", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """Search for images using natural language query"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="No query provided")

    try:
        # Generate text embedding for search
        text_embedding = generate_text_embedding_for_search(request.query)

        # Search in ChromaDB
        results = vector_db.search_by_text(text_embedding, top_k=request.top_k)

        # Convert to response format
        search_results = [
            SearchResult(
                similarity=result["similarity"],
                image_path=result["image_path"],
                metadata=result["metadata"],
                id=result["id"],
            )
            for result in results
        ]

        return SearchResponse(results=search_results, total_results=len(search_results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching images: {str(e)}")


@app.post("/run_embedding", response_model=ProcessingResult)
async def run_embedding_on_uploads():
    """Process all images in uploads folder and store embeddings in ChromaDB"""
    try:
        # Get all image files from uploads directory
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
        image_files = [
            f
            for f in UPLOADS_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            return ProcessingResult(
                message="No images found in uploads folder",
                processed_images=0,
                skipped_images=0,
                total_images=0,
            )

        processed_count = 0
        skipped_count = 0

        for image_file in image_files:
            try:
                # Skip if already in database
                if vector_db.image_exists(str(image_file)):
                    skipped_count += 1
                    continue

                # Load and process image
                image = Image.open(image_file).convert("RGB")
                embedding = generate_image_embedding(image)

                # Create metadata
                metadata = {
                    "original_filename": image_file.name,
                    "file_size": image_file.stat().st_size,
                    "image_format": image.format,
                    "image_size": image.size,
                    "processed_via": "run_embedding",
                }

                # Store in ChromaDB
                vector_db.add_image(str(image_file), embedding, metadata)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                skipped_count += 1

        return ProcessingResult(
            message=f"Successfully processed {processed_count} images, skipped {skipped_count}",
            processed_images=processed_count,
            skipped_images=skipped_count,
            total_images=len(image_files),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing images: {str(e)}"
        )


@app.get("/list_images")
async def list_stored_images():
    """List all images stored in ChromaDB"""
    try:
        images = vector_db.get_all_images()
        return {"total_images": len(images), "images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")


@app.get("/database_stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get statistics about ChromaDB"""
    try:
        total_images = vector_db.get_count()
        return DatabaseStats(
            total_images=total_images,
            database_path=CHROMA_DB_PATH,
            uploads_directory=str(UPLOADS_DIR),
            collection_name=COLLECTION_NAME,
            database_type="ChromaDB",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting database stats: {str(e)}"
        )


@app.delete("/delete_image/{image_id}")
async def delete_image(image_id: str):
    """Delete an image from ChromaDB"""
    try:
        success = vector_db.delete_image(image_id)
        if success:
            return {"message": f"Image {image_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}")


@app.post("/reset_database")
async def reset_database():
    """Reset the ChromaDB collection (delete all data)"""
    try:
        # Delete the collection
        chroma_client.delete_collection(COLLECTION_NAME)

        # Recreate the collection
        global chroma_collection, vector_db
        chroma_collection = chroma_client.create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        vector_db = ChromaVectorDatabase(chroma_collection)

        return {"message": "Database reset successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error resetting database: {str(e)}"
        )


# Expose with ngrok
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)

# Run Uvicorn inside Colab
uvicorn.run(app, host="0.0.0.0", port=8000)
