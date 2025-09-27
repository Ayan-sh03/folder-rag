package orchestrator

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"folder-rag/internal/llm"
	"folder-rag/internal/processor"
	"folder-rag/internal/retriever"
	"folder-rag/internal/watcher"
	"folder-rag/pkg/rag"

	"github.com/fsnotify/fsnotify"
)

type Orchestrator struct {
	cfg       *rag.Config
	watcher   *watcher.Watcher
	processor *processor.Processor
	retriever *retriever.Retriever
	llm       *llm.LLM
	ui        rag.UI
	ctx       context.Context
	cancel    context.CancelFunc
}

func New(cfg rag.Config, w *watcher.Watcher, p *processor.Processor, r *retriever.Retriever, l *llm.LLM, ui rag.UI) *Orchestrator {
	ctx, cancel := context.WithCancel(context.Background())
	return &Orchestrator{
		cfg:       &cfg,
		watcher:   w,
		processor: p,
		retriever: r,
		llm:       l,
		ui:        ui,
		ctx:       ctx,
		cancel:    cancel,
	}
}

func (o *Orchestrator) Start() error {
	defer o.Shutdown()

	// Start background services immediately
	o.ui.PrintWelcome()
	o.ui.PrintEvent("Starting background services", "indexing and file watching")

	// Start event handling goroutine
	go o.handleEvents()

	// Start background file processing
	go o.processExistingFilesBackground()

	// CLI loop starts immediately - no waiting for indexing
	o.runCLI()

	return nil
}

func (o *Orchestrator) processExistingFiles() error {
	return filepath.WalkDir(o.cfg.WatchDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && filepath.Ext(path) == ".txt" {
			o.ui.PrintEvent("Processing existing", path)
			if err := o.processor.ProcessDocument(o.ctx, path); err != nil {
				log.Printf("Error processing %s: %v", path, err)
				o.ui.PrintError(fmt.Sprintf("Failed to process %s: %v", path, err))
			} else {
				o.ui.PrintEvent("Processed", path)
			}
		}
		return nil
	})
}

func (o *Orchestrator) processExistingFilesBackground() {
	var filePaths []string

	// First, collect all .txt files
	err := filepath.WalkDir(o.cfg.WatchDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && filepath.Ext(path) == ".txt" {
			filePaths = append(filePaths, path)
		}
		return nil
	})

	if err != nil {
		log.Printf("Error walking directory: %v", err)
		o.ui.PrintError(fmt.Sprintf("Error scanning directory: %v", err))
		return
	}

	if len(filePaths) == 0 {
		o.ui.PrintEvent("Background indexing complete", "no files found")
		return
	}

	o.ui.PrintEvent("Background indexing started", fmt.Sprintf("found %d files", len(filePaths)))

	// Process files concurrently
	if err := o.processor.ProcessMultipleDocuments(o.ctx, filePaths); err != nil {
		log.Printf("Error in background processing: %v", err)
		o.ui.PrintError(fmt.Sprintf("Background processing error: %v", err))
	} else {
		total, processed := o.processor.GetCacheStats()
		o.ui.PrintEvent("Background indexing complete", fmt.Sprintf("processed %d/%d files", processed, total))
	}
}

func (o *Orchestrator) handleEvents() {
	for {
		select {
		case event := <-o.watcher.Events():
			o.handleEvent(event)
		case err := <-o.watcher.Errors():
			log.Printf("Watcher error: %v", err)
			o.ui.PrintError(fmt.Sprintf("Watcher error: %v", err))
		case <-o.ctx.Done():
			return
		}
	}
}

func (o *Orchestrator) handleEvent(event fsnotify.Event) {
	path := event.Name
	if filepath.Ext(path) != ".txt" {
		return
	}

	eventType := o.getEventType(event.Op)
	o.ui.PrintEvent(eventType, path)

	switch {
	case event.Op&fsnotify.Create == fsnotify.Create || event.Op&fsnotify.Write == fsnotify.Write:
		time.Sleep(100 * time.Millisecond)
		// Process file in background to avoid blocking watcher
		go func(filePath string) {
			if err := o.processor.ProcessDocumentConcurrent(o.ctx, filePath); err != nil {
				o.ui.PrintError(fmt.Sprintf("Process failed for %s: %v", filePath, err))
			} else {
				o.ui.PrintEvent("Processed", filePath)
			}
		}(path)
	case event.Op&fsnotify.Remove == fsnotify.Remove:
		if err := o.processor.DeleteDocument(o.ctx, path); err != nil {
			o.ui.PrintError(fmt.Sprintf("Delete failed for %s: %v", path, err))
		} else {
			o.ui.PrintEvent("Deleted", path)
		}
	case event.Op&fsnotify.Rename == fsnotify.Rename:
		o.ui.PrintEvent("Renamed", path)
		// Reprocess if needed, but log for now
	}
}

func (o *Orchestrator) getEventType(op fsnotify.Op) string {
	switch {
	case op&fsnotify.Create == fsnotify.Create:
		return "Created"
	case op&fsnotify.Write == fsnotify.Write:
		return "Modified"
	case op&fsnotify.Remove == fsnotify.Remove:
		return "Removed"
	case op&fsnotify.Rename == fsnotify.Rename:
		return "Renamed"
	default:
		return "Event"
	}
}

func (o *Orchestrator) runCLI() {
	for {
		query := o.ui.PromptUser()
		if query == "" {
			continue
		}

		// Handle special commands
		switch strings.ToLower(strings.TrimSpace(query)) {
		case "quit", "exit":
			return
		case "clear", "clear history":
			o.llm.ClearHistory()
			o.ui.PrintEvent("History cleared", "")
			continue
		case "history":
			historyLen := o.llm.GetHistoryLength()
			o.ui.PrintEvent("History length", fmt.Sprintf("%d messages", historyLen))
			continue
		case "status", "cache":
			total, processed := o.processor.GetCacheStats()
			o.ui.PrintEvent("Cache status", fmt.Sprintf("%d files cached, %d processed", total, processed))
			continue
		}

		o.ui.PrintProcessing()

		// LLM now handles tool calling and retrieval internally with history
		response, err := o.llm.Generate(o.ctx, query)
		if err != nil {
			o.ui.PrintError(fmt.Sprintf("Error: %v", err))
			continue
		}

		o.ui.PrintResponse(response)
	}
}

func (o *Orchestrator) Shutdown() {
	o.cancel()
	if o.watcher != nil {
		o.watcher.Close()
	}
	// Close other resources if needed
}
