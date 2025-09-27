package rag

import "errors"

type Chunk struct {
	ID       string            `json:"id"`
	Text     string            `json:"text"`
	Vector   []float32         `json:"vector,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type Document struct {
	Path    string  `json:"path"`
	Content string  `json:"content,omitempty"`
	Chunks  []Chunk `json:"chunks,omitempty"`
}

type Config struct {
	// Watching
	WatchDir  string   `json:"watch_dir"`
	WatchDirs []string `json:"watch_dirs,omitempty"` // For multi-dir support

	// Text Processing
	ChunkSize int `json:"chunk_size"`
	Overlap   int `json:"overlap"`

	// Embeddings
	EmbeddingDim   int    `json:"embedding_dim"`
	ReplicateToken string `json:"replicate_token"` // Env: REPLICATE_API_TOKEN
	ModelName      string `json:"model_name"`      // e.g., "sentence-transformers/all-MiniLM-L6-v2"
	ReplicateAPI   string `json:"replicate_api"`
	ModelVersion   string `json:"model_version"`

	// Vector Store
	CollectionName string `json:"collection_name"`
	MilvusAddr     string `json:"milvus_addr"` // Default: "localhost:19530"
	TopK           int    `json:"top_k"`
	ShardNum       int    `json:"shard_num"`

	// LLM
	OpenRouterKey string `json:"openrouter_key"` // Env: OPENROUTER_API_KEY
	LLMModel      string `json:"llm_model"`      // e.g., "openai/gpt-4"
	MaxTokens     int    `json:"max_tokens"`
	OpenRouterURL string `json:"openrouter_url"`

	// General
	BatchSize   int `json:"batch_size"`
	Concurrency int `json:"concurrency"` // For worker pools

	// Defaults will be set in config.Load()
}

type SearchResult struct {
	Text     string  `json:"text"`
	Filename string  `json:"filename"`
	Score    float32 `json:"score"`
}

// LLM Types
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type Tool struct {
	Type     string    `json:"type"`
	Function *Function `json:"function,omitempty"`
}

type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function *struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type ChatResponse struct {
	Choices []struct {
		Message struct {
			Content   string     `json:"content,omitempty"`
			ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
	} `json:"choices"`
}

var (
	ErrInvalidChunk = errors.New("invalid chunk size or overlap")
	ErrNoAPIKey     = errors.New("missing API key")
	// Add more as needed
)
