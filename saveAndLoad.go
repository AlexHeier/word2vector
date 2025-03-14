package main

import (
	"encoding/gob"
	"fmt"
	"os"
)

func (w2v *Word2Vec) SaveModelBinary(outputPath string) {
	file, err := os.Create(outputPath)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(w2v.Vectors)
	if err != nil {
		fmt.Println("Error encoding:", err)
	}
}

func (w2v *Word2Vec) LoadVectorsBinary(filePath string) error {
	// Open the binary file
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filePath, err)
	}
	defer file.Close()

	// Decode the file into the Vectors map
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&w2v.Vectors)
	if err != nil {
		return fmt.Errorf("failed to decode vectors: %v", err)
	}

	// Extract unique words (vocabulary)
	w2v.Vocab = make([]string, 0, len(w2v.Vectors))
	for word := range w2v.Vectors {
		w2v.Vocab = append(w2v.Vocab, word)
	}

	return nil
}
