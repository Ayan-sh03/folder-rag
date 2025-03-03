package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	topK = 5 // number of results to return in search
)

type SearchResult struct {
	Text     string
	Filename string
	Score    float32
}

// queryCollection performs a similarity search in the Milvus collection
func queryCollection(ctx context.Context, milvusClient client.Client, queryText string) ([]SearchResult, error) {
	// Get embedding for query text
	queryEmbedding, err := getEmbedding(queryText)
	if err != nil {
		return nil, fmt.Errorf("error getting query embedding: %v", err)
	}

	logger := log.New(os.Stdout, "INFO: ", log.LstdFlags)
	logger.Println("queryEmbedding: ", queryEmbedding)

	// Prepare search parameters
	sp, err := entity.NewIndexIvfFlatSearchParam(10) // probe 10 clusters
	if err != nil {
		return nil, fmt.Errorf("error creating search parameters: %v", err)
	}
	logger.Println("sp: ", sp)

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

	fmt.Println("results: ", results)

	return results, nil
}

// handleQuery provides an interactive query interface
func handleQuery(ctx context.Context, milvusClient client.Client) {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nEnter your query (or 'quit' to exit): ")
		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)

		if query == "quit" {
			break
		}

		if query == "" {
			continue
		}

		results, err := queryCollection(ctx, milvusClient, query)
		if err != nil {
			fmt.Printf("Error querying collection: %v\n", err)
			continue
		}

		fmt.Println("\nSearch Results:")
		fmt.Println("--------------------")
		for i, result := range results {
			fmt.Printf("\n%d. File: %s (Score: %.4f)\n", i+1, result.Filename, result.Score)
			fmt.Printf("Text: %s\n", result.Text)
		}
	}
}
