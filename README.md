# Vector DB Search API

A FastAPI-based search API that uses Pinecone for vector similarity search.

## Features

- Vector similarity search using Pinecone
- RESTful API endpoints
- CORS enabled for frontend integration
- Environment variable configuration

## Prerequisites

- Python 3.8+
- Pinecone account and API key
- Vercel account (for deployment)

## Local Development

1. Clone the repository
   ```bash
   git clone https://github.com/mujii88/rag_backend.git
   cd rag_backend
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the development server
   ```bash
   uvicorn api.index:app --reload
   ```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /search`: Search endpoint
  ```json
  {
    "query": "your search query",
    "top_k": 3
  }
  ```

## Deployment

### Vercel Deployment

1. Install Vercel CLI (if not installed)
   ```bash
   npm install -g vercel
   ```

2. Login to Vercel
   ```bash
   vercel login
   ```

3. Deploy
   ```bash
   vercel --prod
   ```

## Environment Variables

The following environment variables should be set in your Vercel project settings:

- `PINECONE_API_KEY`: Your Pinecone API key

## Project Structure

```
.
├── api/
│   └── index.py         # Main API entry point
├── main.py              # Local development script
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel configuration
└── .gitignore          # Git ignore file
```

## License

MIT
