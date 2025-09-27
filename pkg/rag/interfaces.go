package rag

import (
	"context"
)

// Embedder generates vector embeddings for text.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	BatchEmbed(ctx context.Context, chunks []Chunk) ([][]float32, error)
}

// VectorStore handles storage, insertion, deletion, and search of vector embeddings.
type VectorStore interface {
	EnsureCollection(ctx context.Context, dim int) error
	Insert(ctx context.Context, chunks []Chunk) error
	DeleteByMetadata(ctx context.Context, key, value string) error
	Search(ctx context.Context, queryVector []float32, topK int, minSimilarity float64) ([]Chunk, error)
	Close() error
}

// LLMProvider generates responses from prompts.
type LLMProvider interface {
	Generate(ctx context.Context, prompt string, maxTokens int) (string, error)
}

// UI handles user interactions and output formatting.
type UI interface {
	PrintEvent(eventType string, path string)
	PrintResponse(response string)
	PromptUser() string
	ConfirmShutdown()

	// Additional methods for full CLI support
	PrintWelcome()
	PrintProcessing()
	PrintError(msg string)
	PrintSearchResults(chunks []Chunk)
}
