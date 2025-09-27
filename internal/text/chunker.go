package text

import (
	"fmt"
	"hash/fnv"

	"folder-rag/pkg/rag"
)

type Chunker struct {
	Size   int
	Overlap int
}

func New(size, overlap int) *Chunker {
	return &Chunker{Size: size, Overlap: overlap}
}

func (c *Chunker) Chunk(content string) []rag.Chunk {
	if len(content) == 0 {
		return nil
	}

	chunks := make([]rag.Chunk, 0)
	for i := 0; i < len(content); i += c.Size - c.Overlap {
		end := i + c.Size
		if end > len(content) {
			end = len(content)
		}
		chunkText := content[i:end]

		// Simple ID: hash of start bytes + index
		h := fnv.New32a()
		h.Write([]byte(chunkText[:min(10, len(chunkText))]))
		id := fmt.Sprintf("%x_%d", h.Sum(nil), i/(c.Size-c.Overlap))

		chunks = append(chunks, rag.Chunk{
			ID:   id,
			Text: chunkText,
		})
	}
	return chunks
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
