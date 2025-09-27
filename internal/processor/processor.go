package processor

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"folder-rag/internal/text"
	"folder-rag/pkg/rag"
)

type FileCache struct {
	Path      string    `json:"path"`
	ModTime   time.Time `json:"mod_time"`
	Size      int64     `json:"size"`
	Hash      string    `json:"hash"`
	Processed bool      `json:"processed"`
}

type Processor struct {
	chunker    *text.Chunker
	embedder   rag.Embedder
	store      rag.VectorStore
	cache      map[string]*FileCache
	cacheMutex sync.RWMutex
	workerPool chan struct{}
}

func New(chunker *text.Chunker, embedder rag.Embedder, store rag.VectorStore) *Processor {
	return &Processor{
		chunker:    chunker,
		embedder:   embedder,
		store:      store,
		cache:      make(map[string]*FileCache),
		workerPool: make(chan struct{}, 5), // 5 concurrent workers
	}
}

func (p *Processor) needsProcessing(path string) (bool, error) {
	stat, err := os.Stat(path)
	if err != nil {
		return false, err
	}

	p.cacheMutex.RLock()
	cached, exists := p.cache[path]
	p.cacheMutex.RUnlock()

	if !exists {
		return true, nil // New file
	}

	// Check if file has been modified
	if stat.ModTime().After(cached.ModTime) || stat.Size() != cached.Size {
		return true, nil // File changed
	}

	return false, nil // File unchanged
}

func (p *Processor) calculateFileHash(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}

	return hex.EncodeToString(hash.Sum(nil)), nil
}

func (p *Processor) updateCache(path string, processed bool) error {
	stat, err := os.Stat(path)
	if err != nil {
		return err
	}

	hash, err := p.calculateFileHash(path)
	if err != nil {
		return err
	}

	p.cacheMutex.Lock()
	p.cache[path] = &FileCache{
		Path:      path,
		ModTime:   stat.ModTime(),
		Size:      stat.Size(),
		Hash:      hash,
		Processed: processed,
	}
	p.cacheMutex.Unlock()

	return nil
}

func (p *Processor) ProcessDocument(ctx context.Context, path string) error {
	// Check if processing is needed
	needsProcessing, err := p.needsProcessing(path)
	if err != nil {
		return err
	}

	if !needsProcessing {
		return nil // Skip processing
	}

	// Delete existing data for this file first (only if file was previously processed)
	p.cacheMutex.RLock()
	cached, exists := p.cache[path]
	p.cacheMutex.RUnlock()

	if exists && cached.Processed {
		if err := p.store.DeleteByMetadata(ctx, "file_path", path); err != nil {
			// Log but don't fail - file might not exist in store
			fmt.Printf("Warning: Could not delete existing data for %s: %v\n", path, err)
		}
	}

	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	chunks := p.chunker.Chunk(string(content))
	if len(chunks) == 0 {
		// Update cache even for empty files
		p.updateCache(path, true)
		return nil
	}

	// Set metadata
	for i := range chunks {
		chunks[i].Metadata = map[string]string{"file_path": path}
	}

	// Embed
	vectors, err := p.embedder.BatchEmbed(ctx, chunks)
	if err != nil {
		return err
	}

	// Assign vectors
	for i := range chunks {
		chunks[i].Vector = vectors[i]
	}

	// Insert
	if err := p.store.Insert(ctx, chunks); err != nil {
		return err
	}

	// Update cache on successful processing
	return p.updateCache(path, true)
}

func (p *Processor) ProcessDocumentConcurrent(ctx context.Context, path string) error {
	// Acquire worker slot
	p.workerPool <- struct{}{}
	defer func() { <-p.workerPool }()

	return p.ProcessDocument(ctx, path)
}

func (p *Processor) ProcessMultipleDocuments(ctx context.Context, paths []string) error {
	errChan := make(chan error, len(paths))
	var wg sync.WaitGroup

	for _, path := range paths {
		wg.Add(1)
		go func(filePath string) {
			defer wg.Done()
			if err := p.ProcessDocumentConcurrent(ctx, filePath); err != nil {
				errChan <- fmt.Errorf("error processing %s: %v", filePath, err)
			}
		}(path)
	}

	wg.Wait()
	close(errChan)

	// Collect any errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("processing errors: %v", errors)
	}

	return nil
}

func (p *Processor) DeleteDocument(ctx context.Context, path string) error {
	// Remove from cache
	p.cacheMutex.Lock()
	delete(p.cache, path)
	p.cacheMutex.Unlock()

	return p.store.DeleteByMetadata(ctx, "file_path", path)
}

func (p *Processor) GetCacheStats() (int, int) {
	p.cacheMutex.RLock()
	defer p.cacheMutex.RUnlock()

	total := len(p.cache)
	processed := 0
	for _, cache := range p.cache {
		if cache.Processed {
			processed++
		}
	}

	return total, processed
}
