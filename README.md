# 🚀 LiteVector AI Database

LiteVector is a highly optimized, in-memory vector database built from scratch to demonstrate the mathematical foundations of AI semantic search. 

Rather than relying on heavy, off-the-shelf database solutions, this project implements a custom vector search engine using hardware-accelerated matrix operations. It is designed to be lightweight, mathematically rigorous, and instantly deployable via Docker.

---

## 🌐 Live Demos

The architecture is fully deployed and accessible via the public web. You can interact with the complete user interface or query the raw backend API directly.

* 🖥️ **[Frontend UI Demo](https://lite-vector-ai-search-engine-lgkw4refl-adarsh-sarathy.vercel.app):** Interact with the live React application, featuring real-time data streaming and a responsive chat interface.
* ⚙️ **[Backend API Swagger UI](https://litevector-database-production.up.railway.app/docs):** Directly query the vector database endpoints and test the search algorithms using the automatically generated OpenAPI documentation.

> **Infrastructure Note:** The backend is hosted on a free-tier Railway container. To conserve resources, the instance spins down after 15 minutes of inactivity. The very first search you run may take 10–15 seconds as the container wakes up and loads the AI model into RAM. All subsequent searches execute in sub-millisecond time.

### 🧠 Core Architecture & Optimizations

* **Vectorized Search Engine:** Utilizes NumPy to execute high-speed matrix multiplications across pre-allocated `float32` arrays, ensuring O(1) insertion time and sub-millisecond query latency.
* **Algorithmic Optimization:** Embeddings are L2-Normalized at the model layer. By locking vector magnitudes to exactly 1, the Cosine Similarity formula collapses into a pure Dot Product, cutting computational load in half by removing the need for magnitude division during search phases.
* **Dynamic Semantic Filtering:** Implements a custom "Delta Drop-off" algorithm at the API layer. Instead of returning a static `top_k` list of results, the system analyzes the relative statistical gap between consecutive similarity scores and automatically truncates the long tail of irrelevant matches.
* **High-Speed I/O Boot Sequence:** Bypasses live AI inference during server boot by loading 4,000 pre-computed document embeddings (sampled from the AG News Dataset) directly from offline-compiled `.npy` binaries, achieving a near-instant startup.

---

### 🛠️ Tech Stack

* **Framework:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** PyTorch, `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Data Structures:** NumPy
* **Infrastructure:** Docker, Docker Compose

---

### 📦 Quick Start (Docker)

The application is fully containerized with multi-stage layer caching optimized for rapid local builds and seamless cloud deployment.

#### 1. Clone the Repository
First, download the source code to your local machine and navigate into the project directory:
```bash
git clone https://github.com/AdarshSarathy/LiteVector-Database.git
cd Vector-Database
```

#### 2. Set Environment Variables (Optional)
Create a `.env` file in the root directory and add your Hugging Face token to enable seamless model caching:
```text
HF_TOKEN=your_huggingface_token
```

#### 3. Boot the Container
Run the following command in your terminal to build the image and launch the database service:
```bash
docker compose up --build
```

#### 4. Test the API
Once the logs show that the 4,000 vectors are loaded into RAM, navigate to the automatically generated Swagger UI to test the endpoints interactively:
[http://localhost:8000/docs](http://localhost:8000/docs)

---

### 📊 Example Queries

The pre-loaded database contains 4,000 news headlines across World, Sports, Business, and Sci/Tech categories. Test the `/search` endpoint using queries such as:
* *"Space exploration and NASA missions"*
* *"Stock market updates and corporate acquisitions"*
* *"Championship sports games and olympics"*
