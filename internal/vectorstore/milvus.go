package vectorstore

import (
	"context"
	"fmt"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"folder-rag/pkg/rag"
)

type MilvusStore struct {
	client client.Client
	cfg    *rag.Config
}

func New(cfg rag.Config) (*MilvusStore, error) {
	ctx := context.Background()
	c, err := client.NewGrpcClient(ctx, cfg.MilvusAddr)
	if err != nil {
		return nil, fmt.Errorf("error creating Milvus client: %v", err)
	}

	store := &MilvusStore{
		client: c,
		cfg:    &cfg,
	}

	if err := store.EnsureCollection(ctx, cfg.EmbeddingDim); err != nil {
		c.Close()
		return nil, err
	}

	return store, nil
}

func (m *MilvusStore) EnsureCollection(ctx context.Context, dim int) error {
	schema := &entity.Schema{
		CollectionName: m.cfg.CollectionName,
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
					"max_length": "65535", // Increased for longer chunks
				},
			},
			{
				Name:     "file_path",
				DataType: entity.FieldTypeVarChar,
				TypeParams: map[string]string{
					"max_length": "1024",
				},
			},
			{
				Name:     "embedding",
				DataType: entity.FieldTypeFloatVector,
				TypeParams: map[string]string{
					"dim": fmt.Sprintf("%d", dim),
				},
			},
		},
	}

	exists, err := m.client.HasCollection(ctx, m.cfg.CollectionName)
	if err != nil {
		return fmt.Errorf("error checking collection existence: %v", err)
	}

	var needsRecreation bool
	if exists {
		// Validate existing schema
		needsRecreation, err = m.validateSchema(ctx, dim)
		if err != nil {
			return fmt.Errorf("error validating schema: %v", err)
		}

		if needsRecreation {
			fmt.Printf("Schema mismatch detected, recreating collection %s\n", m.cfg.CollectionName)
			err = m.client.DropCollection(ctx, m.cfg.CollectionName)
			if err != nil {
				return fmt.Errorf("error dropping collection: %v", err)
			}
			exists = false
		}
	}

	if !exists {
		err = m.client.CreateCollection(ctx, schema, int32(m.cfg.ShardNum))
		if err != nil {
			return fmt.Errorf("error creating collection: %v", err)
		}

		idx, err := entity.NewIndexIvfFlat(entity.L2, 1024)
		if err != nil {
			return fmt.Errorf("error creating index parameters: %v", err)
		}

		err = m.client.CreateIndex(ctx, m.cfg.CollectionName, "embedding", idx, false)
		if err != nil {
			return fmt.Errorf("error creating index: %v", err)
		}
	}

	return nil
}

// validateSchema checks if the existing collection has the required schema
func (m *MilvusStore) validateSchema(ctx context.Context, expectedDim int) (bool, error) {
	collection, err := m.client.DescribeCollection(ctx, m.cfg.CollectionName)
	if err != nil {
		return false, fmt.Errorf("error describing collection: %v", err)
	}

	// Check required fields
	requiredFields := map[string]bool{
		"id":        false,
		"text":      false,
		"file_path": false,
		"embedding": false,
	}

	var embeddingDim int
	for _, field := range collection.Schema.Fields {
		if _, exists := requiredFields[field.Name]; exists {
			requiredFields[field.Name] = true

			// Check embedding dimension
			if field.Name == "embedding" {
				if dimStr, ok := field.TypeParams["dim"]; ok {
					var parsedDim int
					if _, err := fmt.Sscanf(dimStr, "%d", &parsedDim); err == nil {
						embeddingDim = parsedDim
					}
				}
			}
		}
	}

	// Check if all required fields exist
	for fieldName, exists := range requiredFields {
		if !exists {
			fmt.Printf("Missing required field: %s\n", fieldName)
			return true, nil // Needs recreation
		}
	}

	// Check embedding dimension
	if embeddingDim != expectedDim {
		fmt.Printf("Embedding dimension mismatch: expected %d, got %d\n", expectedDim, embeddingDim)
		return true, nil // Needs recreation
	}

	return false, nil // Schema is valid
}

func (m *MilvusStore) Insert(ctx context.Context, chunks []rag.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}

	texts := make([]string, len(chunks))
	filePaths := make([]string, len(chunks))
	embeddings := make([][]float32, len(chunks))

	for i, chunk := range chunks {
		texts[i] = chunk.Text
		filePaths[i] = chunk.Metadata["file_path"]
		embeddings[i] = chunk.Vector
	}

	textColumn := entity.NewColumnVarChar("text", texts)
	filePathColumn := entity.NewColumnVarChar("file_path", filePaths)
	embeddingColumn := entity.NewColumnFloatVector("embedding", m.cfg.EmbeddingDim, embeddings)

	_, err := m.client.Insert(ctx, m.cfg.CollectionName, "", textColumn, filePathColumn, embeddingColumn)
	if err != nil {
		return fmt.Errorf("error inserting into Milvus: %v", err)
	}

	err = m.client.Flush(ctx, m.cfg.CollectionName, false)
	if err != nil {
		return fmt.Errorf("error flushing collection: %v", err)
	}

	return nil
}

func (m *MilvusStore) DeleteByMetadata(ctx context.Context, key, value string) error {
	if key != "file_path" {
		return fmt.Errorf("only file_path metadata deletion supported")
	}

	// Escape backslashes for Windows paths and single quotes
	escapedValue := strings.ReplaceAll(value, "\\", "\\\\")
	escapedValue = strings.ReplaceAll(escapedValue, "'", "\\'")

	expr := fmt.Sprintf("file_path == '%s'", escapedValue)
	err := m.client.Delete(ctx, m.cfg.CollectionName, "", expr)
	if err != nil {
		return fmt.Errorf("error deleting from Milvus: %v", err)
	}

	return m.client.Flush(ctx, m.cfg.CollectionName, false)
}

func (m *MilvusStore) Search(ctx context.Context, queryVector []float32, topK int, minSimilarity float64) ([]rag.Chunk, error) {
	sp, err := entity.NewIndexIvfFlatSearchParam(10)
	if err != nil {
		return nil, fmt.Errorf("error creating search parameters: %v", err)
	}

	err = m.client.LoadCollection(ctx, m.cfg.CollectionName, false)
	if err != nil {
		return nil, fmt.Errorf("error loading collection: %v", err)
	}

	outputFields := []string{"text", "file_path"}
	expr := ""
	vector := []entity.Vector{entity.FloatVector(queryVector)}
	searchResults, err := m.client.Search(
		ctx,
		m.cfg.CollectionName,
		[]string{},
		expr,
		outputFields,
		vector,
		"embedding",
		entity.L2,
		topK,
		sp,
	)
	if err != nil {
		return nil, fmt.Errorf("error searching collection: %v", err)
	}

	var chunks []rag.Chunk
	for idx := 0; idx < topK && idx < len(searchResults) && len(searchResults[0].IDs.(*entity.ColumnInt64).Data()) > idx; idx++ {
		text := searchResults[0].Fields[0].(*entity.ColumnVarChar).Data()[idx]
		filePath := searchResults[0].Fields[1].(*entity.ColumnVarChar).Data()[idx]
		score := searchResults[0].Scores[idx]

		// Filter by minSimilarity if provided (L2 score lower is better, so 1 - normalized or adjust)
		if minSimilarity > 0 && float64(score) > minSimilarity { // Assuming L2, lower better; adjust threshold
			continue
		}

		chunks = append(chunks, rag.Chunk{
			Text: text,
			Metadata: map[string]string{
				"file_path": filePath,
			},
			// ID and Vector not retrieved, but not needed for retrieval
		})
	}

	return chunks, nil
}

func (m *MilvusStore) Close() error {
	return m.client.Close()
}
