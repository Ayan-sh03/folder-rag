package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	chunkSize      = 1000 // characters per chunk
	overlap        = 200  // overlap between chunks
	dimension      = 1024 // embedding dimension
	collectionName = "text_chunks_rag"
	shardNum       = 2 // number of shards for collection
	replicateAPI   = "https://api.replicate.com/v1/predictions"
	modelVersion   = "a06276a89f1a902d5fc225a9ca32b6e8e6292b7f3b136518878da97c458e2bad"
	topK           = 5 // number of results to return in search
)

// ReplicateRequest for embedding API
type ReplicateRequest struct {
	Version string `json:"version"`
	Input   struct {
		Texts               string `json:"texts"`
		BatchSize           int    `json:"batch_size"`
		NormalizeEmbeddings bool   `json:"normalize_embeddings"`
	} `json:"input"`
}

// ReplicateResponse from embedding API
type ReplicateResponse struct {
	ID     string      `json:"id"`
	Status string      `json:"status"`
	Output [][]float32 `json:"output"`
	Error  string      `json:"error,omitempty"`
}

// SearchResult represents a single document match from Milvus
type SearchResult struct {
	Text     string
	Filename string
	Score    float32
}

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

// getEmbedding gets embeddings from Replicate API
func getEmbedding(text string) ([]float32, error) {
	apiToken := os.Getenv("REPLICATE_API_TOKEN")
	if apiToken == "" {
		return nil, fmt.Errorf("REPLICATE_API_TOKEN environment variable not set")
	}
	logger := log.Default()

	// Clean and split the text into sentences
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
	logger.Printf("INFO: Formatted text input: %s", texts)

	reqBody := ReplicateRequest{
		Version: modelVersion,
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

	req, err := http.NewRequest("POST", replicateAPI, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Authorization", "Token "+apiToken)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Prefer", "wait")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	var result ReplicateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("error decoding response: %v", err)
	}

	if result.Error != "" {
		return nil, fmt.Errorf("API error: %s", result.Error)
	}

	if len(result.Output) == 0 || len(result.Output[0]) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return result.Output[0], nil
}

// createCollection creates a Milvus collection if it doesn't exist
func createCollection(ctx context.Context, milvusClient client.Client) error {
	schema := &entity.Schema{
		CollectionName: collectionName,
		Description:    "Text chunks with embeddings",
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeInt64,
				PrimaryKey: true,
				AutoID:     true,
			},
			{
				Name:     "text",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "1024",
				},
			},
			{
				Name:     "filename",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "1024",
				},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", dimension),
				},
			},
		},
	}

	exists, err := milvusClient.HasCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("error checking collection existence: %v", err)
	}

	if !exists {
		err = milvusClient.CreateCollection(ctx, schema, shardNum)
		if err != nil {
			return fmt.Errorf("error creating collection: %v", err)
		}

		idx, err := entity.NewIndexIvfFlat(entity.L2, 1024)
		if err != nil {
			return fmt.Errorf("error creating index parameters: %v", err)
		}

		err = milvusClient.CreateIndex(ctx, collectionName, "embedding", idx, false)
		if err != nil {
			return fmt.Errorf("error creating index: %v", err)
		}
	}

	return nil
}

// processTextFile processes a text file and stores chunks in Milvus
func processTextFile(ctx context.Context, milvusClient client.Client, filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	content, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("error reading file: %v", err)
	}

	text := string(content)
	chunks := make([]string, 0)

	// Create chunks with overlap
	for i := 0; i < len(text); i += chunkSize - overlap {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunk := text[i:end]
		chunks = append(chunks, chunk)
	}

	// Get embeddings for each chunk
	embeddings := make([][]float32, 0)
	texts := make([]string, 0)
	filenames := make([]string, 0)

	for _, chunk := range chunks {
		embedding, err := getEmbedding(chunk)
		if err != nil {
			return fmt.Errorf("error getting embedding: %v", err)
		}

		embeddings = append(embeddings, embedding)
		texts = append(texts, chunk)
		filenames = append(filenames, filepath.Base(filePath))

		// Small delay to avoid rate limiting
		time.Sleep(500 * time.Millisecond)
	}

	// Insert into Milvus
	textColumn := entity.NewColumnVarChar("text", texts)
	filenameColumn := entity.NewColumnVarChar("filename", filenames)
	embeddingColumn := entity.NewColumnFloatVector("embedding", dimension, embeddings)

	_, err = milvusClient.Insert(ctx, collectionName, "", textColumn, filenameColumn, embeddingColumn)
	if err != nil {
		return fmt.Errorf("error inserting into Milvus: %v", err)
	}

	err = milvusClient.Flush(ctx, collectionName, false)
	if err != nil {
		return fmt.Errorf("error flushing collection: %v", err)
	}

	return nil
}

// queryCollection performs a similarity search in the Milvus collection
func queryCollection(ctx context.Context, milvusClient client.Client, queryText string) ([]SearchResult, error) {
	// Get embedding for query text
	queryEmbedding, err := getEmbedding(queryText)
	if err != nil {
		return nil, fmt.Errorf("error getting query embedding: %v", err)
	}

	fmt.Println("Query Embedding:", queryEmbedding)

	// Prepare search parameters
	sp, err := entity.NewIndexIvfFlatSearchParam(10) // probe 10 clusters
	if err != nil {
		return nil, fmt.Errorf("error creating search parameters: %v", err)
	}

	// Load collection if not already loaded
	err = milvusClient.LoadCollection(ctx, collectionName, false)
	if err != nil {
		return nil, fmt.Errorf("error loading collection: %v", err)
	}

	// Prepare search request
	outputFields := []string{"text", "filename"}
	expr := "" // no expression filter
	vector := []entity.Vector{entity.FloatVector(queryEmbedding)}
	searchResult, err := milvusClient.Search(
		ctx,
		collectionName,
		[]string{},   // partition names (empty for all)
		expr,         // filter expression
		outputFields, // output fields to retrieve
		vector,       // search vectors
		"embedding",  // vector field name
		entity.L2,    // metric type
		topK,         // limit
		sp,           // search params
	)

	if err != nil {
		return nil, fmt.Errorf("error searching collection: %v", err)
	}

	// Process results
	var results []SearchResult
	for idx := 0; idx < topK && idx < len(searchResult) && len(searchResult[0].IDs.(*entity.ColumnInt64).Data()) > idx; idx++ {
		text := searchResult[0].Fields[0].(*entity.ColumnVarChar).Data()[idx]
		filename := searchResult[0].Fields[1].(*entity.ColumnVarChar).Data()[idx]
		score := searchResult[0].Scores[idx]
		results = append(results, SearchResult{
			Text:     text,
			Filename: filename,
			Score:    score,
		})
	}

	fmt.Println("Search Results:", results)

	return results, nil
}

// formatMilvusResults formats the search results to provide context to the LLM
func formatMilvusResults(results []SearchResult) string {
	if len(results) == 0 {
		return "No relevant information found in the knowledge base."
	}

	var builder strings.Builder
	builder.WriteString("Here is relevant information from the knowledge base:\n\n")

	for i, result := range results {
		builder.WriteString(fmt.Sprintf("DOCUMENT %d (Source: %s):\n%s\n\n",
			i+1, result.Filename, result.Text))
	}

	return builder.String()
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
	req.Header.Set("HTTP-Referer", "http://localhost")
	req.Header.Set("X-Title", "RAG System")

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

// handleLLMQuery integrates the Milvus query with the LLM tool calling
func handleLLMQuery(ctx context.Context, milvusClient client.Client) {
	reader := bufio.NewReader(os.Stdin)
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set OPENROUTER_API_KEY environment variable")
		return
	}

	// Define query tool for knowledge base
	queryTool := Tool{
		Type: "function",
		Function: &Function{
			Name:        "query_knowledge_base",
			Description: "Query the vector database knowledge base for relevant information",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"question": map[string]string{
						"type":        "string",
						"description": "The question to search for in the knowledge base",
					},
				},
				"required": []string{"question"},
			},
		},
	}

	fmt.Println("\n=== RAG System with LLM ===")
	fmt.Println("This system combines vector search with an LLM to answer your questions.")
	fmt.Println("Type 'quit' to exit.")

	for {
		fmt.Print("\nEnter your question: ")
		userQuery, _ := reader.ReadString('\n')
		userQuery = strings.TrimSpace(userQuery)

		if userQuery == "quit" {
			break
		}

		if userQuery == "" {
			continue
		}

		// Start with user query
		initialMessages := []Message{
			{
				Role:    "user",
				Content: userQuery,
			},
		}

		// Make initial API call to determine if we need to use the knowledge base
		fmt.Println("\nProcessing your question...")
		response, err := makeAPICall(apiKey, initialMessages, []Tool{queryTool})
		if err != nil {
			fmt.Printf("Error in API call: %v\n", err)
			continue
		}

		// Process response
		if len(response.Choices) == 0 {
			fmt.Println("No response from LLM.")
			continue
		}

		// Check if it's a direct response or needs knowledge base lookup
		message := response.Choices[0].Message

		// If no tool calls, just output the content
		if len(message.ToolCalls) == 0 {
			fmt.Println("\nAnswer:", message.Content)
			continue
		}

		// Handle tool calls (knowledge base query)
		fmt.Println("Searching knowledge base...")
		messages := append(initialMessages, Message{
			Role:      "assistant",
			ToolCalls: message.ToolCalls,
		})

		// Process each tool call
		for _, toolCall := range message.ToolCalls {
			if toolCall.Function.Name == "query_knowledge_base" {
				// Parse arguments
				var args struct {
					Question string `json:"question"`
				}
				if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err != nil {
					fmt.Printf("Error parsing arguments: %v\n", err)
					continue
				}

				// Query the Milvus database
				results, err := queryCollection(ctx, milvusClient, args.Question)
				if err != nil {
					fmt.Printf("Error querying knowledge base: %v\n", err)
					continue
				}

				// Format the results
				formattedResults := formatMilvusResults(results)

				// Add results to messages for LLM
				messages = append(messages, Message{
					Role:       "tool",
					Content:    formattedResults,
					ToolCallID: toolCall.ID,
				})
			}
		}

		// Make final API call with knowledge base results
		finalResponse, err := makeAPICall(apiKey, messages, nil)
		if err != nil {
			fmt.Printf("Error in follow-up API call: %v\n", err)
			continue
		}

		// Print final answer
		if len(finalResponse.Choices) > 0 {
			fmt.Println("\nAnswer:", finalResponse.Choices[0].Message.Content)
		}
	}
}

func watchFolder(path string) error {
	// Create Milvus client
	ctx := context.Background()
	milvusClient, err := client.NewGrpcClient(ctx, "localhost:19530")
	if err != nil {
		return fmt.Errorf("error creating Milvus client: %v", err)
	}
	defer milvusClient.Close()

	// Create collection
	err = createCollection(ctx, milvusClient)
	if err != nil {
		return fmt.Errorf("error creating collection: %v", err)
	}

	// Create new watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return fmt.Errorf("error creating watcher: %v", err)
	}
	defer watcher.Close()
	fmt.Printf("Watching directory: %s\n", path)

	// Start listening for events
	go func() {
		for {
			select {
			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				switch {
				case event.Has(fsnotify.Create):
					fmt.Printf("\nCreated: %s\n", event.Name)
					if filepath.Ext(event.Name) == ".txt" {
						fmt.Printf("Processing text file: %s\n", event.Name)

						// Add small delay to ensure file is fully written
						time.Sleep(100 * time.Millisecond)

						if err := processTextFile(ctx, milvusClient, event.Name); err != nil {
							fmt.Printf("Error processing file: %v\n", err)
						} else {
							fmt.Printf("Successfully processed file: %s\n", event.Name)
						}
					}
					printFileDetails(event.Name)
				case event.Has(fsnotify.Write):
					fmt.Printf("Modified: %s\n", event.Name)
				case event.Has(fsnotify.Remove):
					fmt.Printf("Removed: %s\n", event.Name)
				case event.Has(fsnotify.Rename):
					fmt.Printf("Renamed: %s\n", event.Name)
				case event.Has(fsnotify.Chmod):
					fmt.Printf("Permissions modified: %s\n", event.Name)
				}
			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				fmt.Printf("Error: %v\n", err)
			}
		}
	}()

	err = watcher.Add(path)
	if err != nil {
		return fmt.Errorf("error watching directory: %v", err)
	}

	fmt.Printf("Started watching directory: %s\n", path)
	fmt.Println("Press Ctrl+C to stop...")

	// Use the new integrated LLM query handler instead of the old one
	handleLLMQuery(ctx, milvusClient)

	<-make(chan struct{})
	return nil
}

func printFileDetails(path string) {
	file, err := os.Stat(path)
	if err != nil {
		fmt.Printf("Error getting file details: %v\n", err)
		return
	}

	fmt.Printf("\nFile Details for: %s\n", path)
	fmt.Printf("Size: %d bytes\n", file.Size())
	fmt.Printf("Modified: %s\n", file.ModTime().Format(time.RFC822))
	fmt.Printf("Permissions: %s\n", file.Mode().String())

	if filepath.Ext(path) == ".txt" {
		f, err := os.Open(path)
		if err != nil {
			fmt.Printf("Error opening file: %v\n", err)
			return
		}
		defer f.Close()

		scanner := bufio.NewScanner(f)
		if scanner.Scan() {
			fmt.Printf("First line: %s\n", scanner.Text())
		} else if err := scanner.Err(); err != nil {
			fmt.Printf("Error reading file: %v\n", err)
		}
	}
}

func main() {
	fmt.Println("Starting RAG system with LLM integration...")
	dir := "." // Default to test directory
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}
	if err := watchFolder(dir); err != nil {
		log.Fatal(err)
	}
}
