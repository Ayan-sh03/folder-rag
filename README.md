# RAG System with LLM Integration

A Real-time Document Processing and Question-Answering System that combines vector search capabilities with LLM (Language Model) integration for intelligent information retrieval.

## Features

- 📝 Real-time text file monitoring and processing
- 🔍 Vector-based semantic search using Milvus
- 🤖 LLM integration for intelligent responses
- 🎨 Beautiful colored output interface
- 📊 Automatic text chunking with overlap
- 🔄 Real-time file system event handling

## Prerequisites

- Go 1.19 or later
- Docker and Docker Compose (for running Milvus)
- Replicate API key for embeddings generation
- OpenRouter API key for LLM integration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ayan-sh03/folder-rag.git
cd folder-rag
```

2. Install dependencies:
```bash
go mod tidy
```

3. Start Milvus using Docker Compose:
```bash
docker-compose up -d
```

## Configuration

The system requires two environment variables:

1. `REPLICATE_API_TOKEN`: Your Replicate API token for generating embeddings
2. `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM access

Set these environment variables before running the application:

```bash
export REPLICATE_API_TOKEN='your-replicate-token'
export OPENROUTER_API_KEY='your-openrouter-key'
```

## Usage

1. Start the application:
```bash
go run main.go [directory-path]
```
If no directory path is provided, it will watch the current directory.

2. The system will:
   - Monitor the specified directory for new text files
   - Process any new .txt files automatically
   - Provide a command-line interface for asking questions

3. Interacting with the system:
   - Type your questions when prompted
   - The system will search the knowledge base and provide relevant answers
   - Type 'quit' to exit

## Text Processing Details

- Chunk Size: 1000 characters
- Chunk Overlap: 200 characters
- Embedding Dimension: 1024
- Vector Search: Top 5 results

## File Event Handling

The system monitors and responds to the following file events:
- ✨ Creation of new files
- 📝 Modification of existing files
- 🗑️ Removal of files
- 🔄 Renaming of files
- 📋 Permission changes

## Features in Detail

### Vector Search
- Uses Milvus for efficient vector storage and similarity search
- Implements IVF (Inverted File) index with L2 distance metric
- Supports real-time updates and queries

### LLM Integration
- Uses OpenRouter's API for accessing advanced language models
- Implements function calling for structured knowledge base queries
- Provides context-aware responses based on retrieved documents

### Text Processing
- Automatic text chunking with configurable overlap
- Embedding generation using Replicate's API
- Efficient batch processing with rate limiting

### Visual Interface
- Colored output for different types of information
- Boxed messages for better readability
- Clear status indicators and error messages

## Error Handling

The system includes comprehensive error handling for:
- API failures (both Replicate and OpenRouter)
- File system operations
- Database operations
- Invalid user input

## Performance Considerations

- Uses goroutines for non-blocking file system monitoring
- Implements rate limiting for API calls
- Efficient vector indexing for fast similarity search
- Batch processing of document chunks

## Contributing

Contributions are welcome! Please feel free to submit pull requests.
