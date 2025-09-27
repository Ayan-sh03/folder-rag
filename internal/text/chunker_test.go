package text

import "testing"

func TestChunkerEmptyContent(t *testing.T) {
chunker := New(100, 10)
if chunks := chunker.Chunk(""); len(chunks) != 0 {
t.Fatalf("expected 0 chunks, got %d", len(chunks))
}
}

func TestChunkerCreatesOverlappingChunks(t *testing.T) {
chunker := New(5, 2)
content := "abcdefghij"

chunks := chunker.Chunk(content)
if len(chunks) != 3 {
t.Fatalf("expected 3 chunks, got %d", len(chunks))
}

expected := []string{"abcde", "defgh", "ghij"}
for i, chunk := range chunks {
if chunk.Text != expected[i] {
t.Fatalf("chunk %d text mismatch: expected %q, got %q", i, expected[i], chunk.Text)
}
if chunk.ID == "" {
t.Fatalf("chunk %d has empty ID", i)
}
}
}
