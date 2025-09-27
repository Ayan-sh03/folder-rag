package llm

import (
	"context"

	"folder-rag/internal/retriever"
	"folder-rag/pkg/rag"
)

type LLM struct {
	provider  *OpenRouterProvider
	retriever *retriever.Retriever
	maxTokens int
	history   []rag.Message
}

func New(provider *OpenRouterProvider, retriever *retriever.Retriever, maxTokens int) *LLM {
	return &LLM{
		provider:  provider,
		retriever: retriever,
		maxTokens: maxTokens,
		history:   make([]rag.Message, 0),
	}
}

func (l *LLM) Generate(ctx context.Context, query string) (string, error) {
	response, err := l.provider.GenerateWithHistory(ctx, query, l.maxTokens, l.retriever, l.history)
	if err != nil {
		return "", err
	}

	// Add user query and assistant response to history
	l.addToHistory(rag.Message{Role: "user", Content: query})
	l.addToHistory(rag.Message{Role: "assistant", Content: response})

	return response, nil
}

func (l *LLM) addToHistory(message rag.Message) {
	l.history = append(l.history, message)
	l.manageHistoryLength()
}

func (l *LLM) manageHistoryLength() {
	const maxHistoryMessages = 20 // Keep last 20 messages (10 exchanges)

	if len(l.history) > maxHistoryMessages {
		// Keep the most recent messages
		l.history = l.history[len(l.history)-maxHistoryMessages:]
	}
}

func (l *LLM) ClearHistory() {
	l.history = make([]rag.Message, 0)
}

func (l *LLM) GetHistoryLength() int {
	return len(l.history)
}
