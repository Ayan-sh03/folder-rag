package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
)

// Message represents a chat message
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Tool represents a function that the AI can call
type Tool struct {
	Type     string    `json:"type"`
	Function *Function `json:"function,omitempty"`
}

// Function represents the details of a callable function
type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ChatRequest represents the request payload to the API
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

// ToolCall represents a tool call in the response
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function *struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ChatResponse represents the API response
type ChatResponse struct {
	Choices []struct {
		Message struct {
			Content   string     `json:"content,omitempty"`
			ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
	} `json:"choices"`
}

func initilise() {
	// Get API key from environment variable
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENROUTER_API_KEY environment variable")
		return
	}

	// Define tools
	weatherTool := Tool{
		Type: "function",
		Function: &Function{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"location": map[string]string{
						"type":        "string",
						"description": "The city and country (e.g., 'New York, USA')",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	queryTool := Tool{
		Type: "function",
		Function: &Function{
			Name:        "query",
			Description: "Query the knowledge base for the given question",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"question": map[string]string{
						"type":        "string",
						"description": "The question to be asked to the knowledge base",
					},
				},
				"required": []string{"question"},
			},
		},
	}

	// Initial request
	initialMessages := []Message{
		{
			Role:    "user",
			Content: "What's the capital of France?",
		},
	}

	// Make first API call
	response, err := makeAPICall(apiKey, initialMessages, []Tool{queryTool})
	if err != nil {
		fmt.Printf("Error in initial API call: %v\n", err)
		return
	}

	// Process first response
	if len(response.Choices) > 0 {
		message := response.Choices[0].Message
		fmt.Println("=== Initial Response ===")
		if message.Content != "" {
			fmt.Println("Content:", message.Content)
		}

		// Check if it's a direct response
		if message.Content != "" && len(message.ToolCalls) == 0 {
			return
		}

		// Handle tool calls
		if len(message.ToolCalls) > 0 {
			fmt.Println("Tool Calls Detected:")
			messages := append(initialMessages, Message{
				Role:      "assistant",
				ToolCalls: message.ToolCalls,
			})

			// Process each tool call
			for i, toolCall := range message.ToolCalls {
				fmt.Printf("- Tool Call %d: %s\n", i+1, toolCall.Function.Name)
				if toolCall.Function.Name == "query" {
					// Parse arguments
					var args struct {
						Question string `json:"question"`
					}
					if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
						fmt.Printf("  Error parsing arguments: %v\n", err)
						continue
					}

					// Execute tool
					result := query(args.Question)
					fmt.Printf("  Result: %s\n", result)

					// Add tool result to messages
					messages = append(messages, Message{
						Role:       "tool",
						Content:    result,
						ToolCallID: toolCall.ID,
					})
				}
			}

			// Make second API call with tool results
			fmt.Println("\n=== Follow-Up Request ===")
			finalResponse, err := makeAPICall(apiKey, messages, []Tool{weatherTool})
			if err != nil {
				fmt.Printf("Error in follow-up API call: %v\n", err)
				return
			}

			// Print final response
			if len(finalResponse.Choices) > 0 {
				fmt.Println("=== Final Response ===")
				fmt.Println("Content:", finalResponse.Choices[0].Message.Content)
			}
		}
	}
}

// makeAPICall handles the HTTP request to the OpenRouter API
func makeAPICall(apiKey string, messages []Message, tools []Tool) (*ChatResponse, error) {
	url := "https://openrouter.ai/api/v1/chat/completions"

	requestBody := ChatRequest{
		Model:    "google/gemini-2.0-flash-001", // Model with tool calling support
		Messages: messages,
		Tools:    tools,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling JSON: %v", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("HTTP-Referer", "http://localhost") // Replace with your app's URL
	req.Header.Set("X-Title", "Tool Calling Example")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response: %v", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var chatResponse ChatResponse
	err = json.Unmarshal(body, &chatResponse)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling response: %v", err)
	}

	return &chatResponse, nil
}

// getWeather is a placeholder tool implementation
func getWeather(location string) string {
	return fmt.Sprintf("Weather for %s: Sunny, 72°F", location)
}

// query is a placeholder tool implementation
func query(question string) string {
	if question == "What's the capital of France?" {
		return "Paris"
	}
	return "I'm sorry, I don't know the answer to that question"
}
