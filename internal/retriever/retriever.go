package retriever

import (
	"context"

	"folder-rag/pkg/rag"
)

type Retriever struct {
	embedder rag.Embedder
	store    rag.VectorStore
	topK     int
}

func New(embedder rag.Embedder, store rag.VectorStore, topK int) *Retriever {
	return &Retriever{
		embedder: embedder,
		store:    store,
		topK:     topK,
	}
}

func (r *Retriever) Retrieve(ctx context.Context, query string) ([]rag.Chunk, error) {
	queryVec, err := r.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}

	return r.store.Search(ctx, queryVec, r.topK, 0.0) // No min similarity filter
}
