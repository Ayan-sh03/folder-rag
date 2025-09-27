package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"folder-rag/internal/retriever"
	"folder-rag/pkg/rag"
)

type OpenRouterProvider struct {
	key    string
	url    string
	model  string
	client *http.Client
}

func NewOpenRouter(cfg rag.Config) *OpenRouterProvider {
	return &OpenRouterProvider{
		key:    cfg.OpenRouterKey,
		url:    cfg.OpenRouterURL,
		model:  cfg.LLMModel,
		client: &http.Client{},
	}
}

func (o *OpenRouterProvider) Generate(ctx context.Context, prompt string, maxTokens int, retriever *retriever.Retriever) (string, error) {
	return o.GenerateWithHistory(ctx, prompt, maxTokens, retriever, []rag.Message{})
}

func (o *OpenRouterProvider) GenerateWithHistory(ctx context.Context, prompt string, maxTokens int, retriever *retriever.Retriever, history []rag.Message) (string, error) {
	// Define query tool for knowledge base
	queryTool := rag.Tool{
		Type: "function",
		Function: &rag.Function{
			Name:        "query_knowledge_base",
			Description: "Query the vector database knowledge base for relevant information",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"question": map[string]interface{}{
						"type":        "string",
						"description": "The question to search for in the knowledge base",
					},
				},
				"required": []string{"question"},
			},
		},
	}

	// Build messages with history + current prompt
	messages := make([]rag.Message, 0, len(history)+2)

	// Add system prompt if this is the first message or no system prompt exists
	hasSystemPrompt := len(history) > 0 && history[0].Role == "system"
	if !hasSystemPrompt {
		messages = append(messages, rag.Message{
			Role: "system",
			Content: "You are a helpful AI assistant with access to a knowledge base and conversation history. " +
				"You can remember previous questions and responses from this conversation. " +
				"When asked about previous prompts, questions, or our conversation history, " +
				"refer to the message history provided in this conversation.",
		})
	}

	messages = append(messages, history...)
	messages = append(messages, rag.Message{
		Role:    "user",
		Content: prompt,
	})

	// Make initial API call to determine if we need to use the knowledge base
	response, err := o.makeAPICall(ctx, messages, []rag.Tool{queryTool})
	if err != nil {
		return "", err
	}

	// Process response
	if len(response.Choices) == 0 {
		return "No response from LLM.", nil
	}

	message := response.Choices[0].Message

	// If no tool calls, just return the content
	if len(message.ToolCalls) == 0 {
		return message.Content, nil
	}

	// Handle tool calls (knowledge base query)
	messages = append(messages, rag.Message{
		Role:      "assistant",
		ToolCalls: message.ToolCalls,
	})

	// Process each tool call
	for _, toolCall := range message.ToolCalls {
		if toolCall.Function.Name == "query_knowledge_base" {
			var args struct {
				Question string `json:"question"`
			}
			if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
				return "", fmt.Errorf("error parsing tool arguments: %v", err)
			}

			// Query the knowledge base
			chunks, err := retriever.Retrieve(ctx, args.Question)
			if err != nil {
				return "", fmt.Errorf("error querying knowledge base: %v", err)
			}

			// Format the results
			formattedResults := o.formatResults(chunks)

			// Add results to messages for LLM
			messages = append(messages, rag.Message{
				Role:       "tool",
				Content:    formattedResults,
				ToolCallID: toolCall.ID,
			})
		}
	}

	// Make final API call with knowledge base results
	finalResponse, err := o.makeAPICall(ctx, messages, nil)
	if err != nil {
		return "", err
	}

	// Return final answer
	if len(finalResponse.Choices) > 0 {
		return finalResponse.Choices[0].Message.Content, nil
	}

	return "No response generated.", nil
}

func (o *OpenRouterProvider) makeAPICall(ctx context.Context, messages []rag.Message, tools []rag.Tool) (*rag.ChatResponse, error) {
	requestBody := rag.ChatRequest{
		Model:    o.model,
		Messages: messages,
		Tools:    tools,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %v", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", o.url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+o.key)
	httpReq.Header.Set("HTTP-Referer", "http://localhost")
	httpReq.Header.Set("X-Title", "RAG System")

	resp, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(body))
	}

	var chatResp rag.ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %v", err)
	}

	return &chatResp, nil
}

func (o *OpenRouterProvider) formatResults(chunks []rag.Chunk) string {
	if len(chunks) == 0 {
		return "No relevant information found in the knowledge base."
	}

	var builder strings.Builder
	builder.WriteString("Here is relevant information from the knowledge base:\n\n")

	for i, chunk := range chunks {
		filePath := "unknown"
		if p, ok := chunk.Metadata["file_path"]; ok {
			filePath = p
		}
		builder.WriteString(fmt.Sprintf("DOCUMENT %d (Source: %s):\n%s\n\n",
			i+1, filePath, chunk.Text))
	}

	return builder.String()
}
