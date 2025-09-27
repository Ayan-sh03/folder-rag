package main

import (
	"folder-rag/internal/config"
	"folder-rag/internal/embedder"
	"folder-rag/internal/llm"
	"folder-rag/internal/orchestrator"
	"folder-rag/internal/processor"
	"folder-rag/internal/retriever"
	"folder-rag/internal/text"
	"folder-rag/internal/ui"
	"folder-rag/internal/vectorstore"
	"folder-rag/internal/watcher"
	"log"
)

func main() {
	cfg := config.Load()

	// Create components
	chunker := text.New(cfg.ChunkSize, cfg.Overlap)
	embedderInst := embedder.New(cfg)
	store, err := vectorstore.New(cfg)
	if err != nil {
		log.Fatal("Failed to create vector store: ", err)
	}
	defer store.Close()

	proc := processor.New(chunker, embedderInst, store)
	retr := retriever.New(embedderInst, store, cfg.TopK)
	provider := llm.NewOpenRouter(cfg)
	llmInst := llm.New(provider, retr, cfg.MaxTokens) // Pass retriever for tool calling
	userInterface := ui.New()

	w, err := watcher.New(cfg.WatchDir)
	if err != nil {
		log.Fatal("Failed to create watcher: ", err)
	}
	defer w.Close()

	orch := orchestrator.New(cfg, w, proc, retr, llmInst, userInterface)

	if err := orch.Start(); err != nil {
		log.Fatal("Orchestrator error: ", err)
	}
}
