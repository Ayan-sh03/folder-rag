package ui

import (
	"bufio"
	"fmt"
	"folder-rag/pkg/rag"
	"os"
	"strings"
)

type UI struct{}

func New() rag.UI {
	return &UI{}
}

const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorBlue   = "\033[34m"
	colorPurple = "\033[35m"
	colorCyan   = "\033[36m"

	boxTopLeft     = "╭"
	boxTopRight    = "╮"
	boxBottomLeft  = "╰"
	boxBottomRight = "╯"
	boxHorizontal  = "─"
	boxVertical    = "│"
)

func (u *UI) drawBox(text, color string) string {
	lines := strings.Split(text, "\n")
	maxWidth := 0
	for _, line := range lines {
		if len(line) > maxWidth {
			maxWidth = len(line)
		}
	}

	var result strings.Builder
	// Top border
	result.WriteString(color + boxTopLeft + strings.Repeat(boxHorizontal, maxWidth+2) + boxTopRight + colorReset + "\n")

	// Content
	for _, line := range lines {
		result.WriteString(color + boxVertical + " " + line + strings.Repeat(" ", maxWidth-len(line)) + " " + boxVertical + colorReset + "\n")
	}

	// Bottom border
	result.WriteString(color + boxBottomLeft + strings.Repeat(boxHorizontal, maxWidth+2) + boxBottomRight + colorReset)
	return result.String()
}

func (u *UI) PrintEvent(eventType, path string) {
	color := u.getColor(eventType)
	fmt.Printf("%sEvent: %s %s%s\n", color, eventType, path, colorReset)
}

func (u *UI) getColor(eventType string) string {
	switch strings.ToLower(eventType) {
	case "created", "processed":
		return colorGreen
	case "modified":
		return colorYellow
	case "removed", "deleted":
		return colorRed
	case "renamed":
		return colorPurple
	default:
		return colorCyan
	}
}

func (u *UI) PrintResponse(response string) {
	fmt.Println(u.drawBox(response, colorPurple))
}

func (u *UI) PromptUser() string {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print(colorGreen + "Enter your question: " + colorReset)
	query, _ := reader.ReadString('\n')
	return strings.TrimSpace(query)
}

func (u *UI) ConfirmShutdown() {
	fmt.Println(colorYellow + "Shutting down gracefully..." + colorReset)
}

func (u *UI) PrintWelcome() {
	welcome := "=== RAG System with LLM ===\nThis system combines vector search with an LLM to answer your questions.\nType 'quit' to exit."
	fmt.Println(u.drawBox(welcome, colorBlue))
}

func (u *UI) PrintProcessing() {
	fmt.Println(u.drawBox("Processing your question...", colorCyan))
}

func (u *UI) PrintError(msg string) {
	fmt.Println(u.drawBox(fmt.Sprintf("Error: %s", msg), colorRed))
}

func (u *UI) PrintSearchResults(chunks []rag.Chunk) {
	if len(chunks) == 0 {
		fmt.Println(u.drawBox("No relevant context found.", colorYellow))
		return
	}
	fmt.Println(u.drawBox("Found relevant context:", colorYellow))
	for i, chunk := range chunks {
		filePath := "unknown"
		if p, ok := chunk.Metadata["file_path"]; ok {
			filePath = p
		}
		preview := chunk.Text
		if len(preview) > 200 {
			preview = preview[:200] + "..."
		}
		fmt.Printf("%d. From %s:\n%s\n\n", i+1, filePath, preview)
	}
}
