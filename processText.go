package main

import (
	"bufio"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

func (w2v *Word2Vec) preprocessText(folderPath string) (allWords []string, err error) {
	uniqueWords := []string{}
	// Initialize a map to track unique words from Vocab for faster lookup
	vocabSet := make(map[string]struct{})
	for _, word := range w2v.Vocab {
		vocabSet[word] = struct{}{} // Populate vocabSet with words from Vocab
	}

	// Loop through all .txt files in the folder
	err = filepath.Walk(folderPath, func(filePath string, info os.FileInfo, err error) error {
		// Skip directories and non .txt files
		if err != nil || info.IsDir() || filepath.Ext(filePath) != ".txt" {
			return nil
		}

		// Open the file
		file, err := os.Open(filePath)
		if err != nil {
			return err
		}
		defer file.Close()

		// Regular expression to remove punctuation
		re := regexp.MustCompile(`[^\w\s]`)

		// Create a scanner to read the file line by line
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()

			// Clean the line: remove punctuation, convert to lowercase, and trim spaces
			line = re.ReplaceAllString(line, "")
			line = strings.ToLower(line)
			line = strings.TrimSpace(line)

			// Tokenize the cleaned line into words
			for _, word := range strings.Fields(line) {
				if word != "" {
					// Append to allWords slice
					allWords = append(allWords, word)

					// Check if the word is already in Vocab (using vocabSet for fast lookup)
					if _, exists := vocabSet[word]; !exists {
						// If it's not in vocab, add it to uniqueWords and vocabSet
						vocabSet[word] = struct{}{}
						uniqueWords = append(uniqueWords, word)
					}
				}
			}
		}

		// Check for errors while scanning the file
		if err := scanner.Err(); err != nil {
			return err
		}
		return nil
	})

	// Return any errors encountered during the folder scan
	if err != nil {
		return nil, err
	}

	w2v.AddUniqueWords(uniqueWords)

	return allWords, nil
}

/*
Checks if words from an array of words are within the vocabulary. If not, adds them to the vocabulary and initializes their vectors.
*/
func (w2v *Word2Vec) AddUniqueWords(uniqueWords []string) {
	for _, word := range uniqueWords {
		if _, exists := w2v.Vectors[word]; !exists {
			w2v.Vocab = append(w2v.Vocab, word)
			vector := make([]float64, vectorSize)
			for i := 0; i < vectorSize; i++ {
				vector[i] = rand.Float64() * 0.1
			}
			w2v.Vectors[word] = vector
		}
	}
}
