package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"folder-rag/pkg/rag"
)

type ReplicateEmbedder struct {
	apiURL       string
	token        string
	modelVersion string
	client       *http.Client
}

type ReplicateRequest struct {
	Version string `json:"version"`
	Input   struct {
		Texts               string `json:"texts"`
		BatchSize           int    `json:"batch_size"`
		NormalizeEmbeddings bool   `json:"normalize_embeddings"`
	} `json:"input"`
}

type ReplicateResponse struct {
	ID     string      `json:"id"`
	Status string      `json:"status"`
	Output [][]float32 `json:"output"`
	Error  string      `json:"error,omitempty"`
}

func New(cfg rag.Config) *ReplicateEmbedder {
	return &ReplicateEmbedder{
		apiURL:       cfg.ReplicateAPI,
		token:        cfg.ReplicateToken,
		modelVersion: cfg.ModelVersion,
		client:       &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *ReplicateEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	chunks := []rag.Chunk{{Text: text}}
	embeddings, err := e.BatchEmbed(ctx, chunks)
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embedding generated")
	}
	return embeddings[0], nil
}

func (e *ReplicateEmbedder) BatchEmbed(ctx context.Context, chunks []rag.Chunk) ([][]float32, error) {
	if len(chunks) == 0 {
		return nil, nil
	}

	// Format texts as JSON array of strings (Replicate expects array of texts)
	var texts []string
	for _, chunk := range chunks {
		// Clean text: split into sentences if multi-line, but for simplicity, treat whole as one
		cleaned := strings.ReplaceAll(chunk.Text, "\r", "")
		sentences := strings.Split(cleaned, "\n")
		var valid []string
		for _, s := range sentences {
			s = strings.TrimSpace(s)
			if s != "" {
				escaped := strings.ReplaceAll(s, `"`, `\"`)
				valid = append(valid, fmt.Sprintf(`"%s"`, escaped))
			}
		}
		if len(valid) > 0 {
			texts = append(texts, "["+strings.Join(valid, ",")+"]")
		} else {
			texts = append(texts, `" "`) // Empty text fallback
		}
	}

	// Since API takes one 'texts' array, but for batch chunks, we need to call per chunk or adjust.
	// Original: API call per chunk, but to batch, we can make one call with all sentences from all chunks, but metadata lost.
	// Better: Call API once per chunk for now, but sequential with delay. For true batch, parallel goroutines.
	// To optimize, use goroutines for concurrent calls, limit concurrency.

	embeddings := make([][]float32, len(chunks))
	errChan := make(chan error, len(chunks))
	var wg sync.WaitGroup

	sem := make(chan struct{}, 5) // Limit 5 concurrent API calls

	for i, chunk := range chunks {
		wg.Add(1)
		go func(idx int, ch rag.Chunk) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			vec, err := e.embedSingle(ctx, ch.Text)
			if err != nil {
				errChan <- err
				return
			}
			embeddings[idx] = vec
			time.Sleep(500 * time.Millisecond) // Rate limit
		}(i, chunk)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		if err != nil {
			return nil, err // Or collect errors
		}
	}

	return embeddings, nil
}

func (e *ReplicateEmbedder) embedSingle(ctx context.Context, text string) ([]float32, error) {
	// Same as original getEmbedding
	cleanedText := strings.ReplaceAll(text, "\r", "")
	sentences := strings.Split(cleanedText, "\n")
	var validSentences []string
	for _, s := range sentences {
		s = strings.TrimSpace(s)
		if s != "" {
			escaped := strings.ReplaceAll(s, `"`, `\"`)
			validSentences = append(validSentences, fmt.Sprintf(`"%s"`, escaped))
		}
	}
	texts := "[" + strings.Join(validSentences, ",") + "]"

	reqBody := ReplicateRequest{
		Version: e.modelVersion,
		Input: struct {
			Texts               string `json:"texts"`
			BatchSize           int    `json:"batch_size"`
			NormalizeEmbeddings bool   `json:"normalize_embeddings"`
		}{
			Texts:               texts,
			BatchSize:           32,
			NormalizeEmbeddings: true,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.apiURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Authorization", "Token "+e.token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Prefer", "wait")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	var result ReplicateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("error decoding response: %v, body: %s", err, string(body))
	}

	if result.Error != "" {
		return nil, fmt.Errorf("API error: %s", result.Error)
	}

	if len(result.Output) == 0 || len(result.Output[0]) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return result.Output[0], nil // Return first embedding (for the text)
}
