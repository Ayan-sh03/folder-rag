package config

import (
	"flag"
	"os"

	"errors"
	"folder-rag/pkg/rag"
)

func Load() rag.Config {
	cfg := rag.Config{
		WatchDir:       ".",
		ChunkSize:      1000,
		Overlap:        200,
		EmbeddingDim:   1024,
		CollectionName: "text_chunks_rag",
		MilvusAddr:     "localhost:19530",
		TopK:           5,
		MaxTokens:      1000,
		BatchSize:      10,
		Concurrency:    5,
		ModelName:      "sentence-transformers/all-MiniLM-L6-v2", // Default embedding model
		LLMModel:       "deepseek/deepseek-chat-v3.1:free",       // Updated from main.go
		ReplicateAPI:   "https://api.replicate.com/v1/predictions",
		ModelVersion:   "a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad",
		ShardNum:       2,
		OpenRouterURL:  "https://openrouter.ai/api/v1/chat/completions",
	}

	// CLI flags for overrides
	flag.StringVar(&cfg.WatchDir, "watch-dir", cfg.WatchDir, "Directory to watch for files")
	flag.IntVar(&cfg.ChunkSize, "chunk-size", cfg.ChunkSize, "Chunk size in characters")
	flag.IntVar(&cfg.Overlap, "overlap", cfg.Overlap, "Chunk overlap in characters")
	flag.StringVar(&cfg.MilvusAddr, "milvus-addr", cfg.MilvusAddr, "Milvus server address")
	flag.IntVar(&cfg.TopK, "top-k", cfg.TopK, "Number of top results to retrieve")
	flag.StringVar(&cfg.CollectionName, "collection-name", cfg.CollectionName, "Milvus collection name")
	flag.StringVar(&cfg.LLMModel, "llm-model", cfg.LLMModel, "LLM model name")
	flag.Parse()

	// Env vars
	cfg.ReplicateToken = os.Getenv("REPLICATE_API_TOKEN")
	cfg.OpenRouterKey = os.Getenv("OPENROUTER_API_KEY")

	// Validation
	if cfg.ReplicateToken == "" || cfg.OpenRouterKey == "" {
		panic(rag.ErrNoAPIKey)
	}
	if cfg.ChunkSize <= cfg.Overlap {
		panic(rag.ErrInvalidChunk)
	}
	if cfg.EmbeddingDim <= 0 {
		panic(errors.New("invalid embedding dimension"))
	}

	return cfg
}
