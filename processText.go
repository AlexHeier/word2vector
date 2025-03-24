package main

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

func (w2v *Word2Vec) preprocessText(folderPath string) (allWords []string, err error) {
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

			line = strings.ReplaceAll(line, "\n", " ")
			line = strings.ReplaceAll(line, "\r", " ")
			line = re.ReplaceAllString(line, "")
			line = strings.ToLower(line)
			line = strings.TrimSpace(line)

			// Tokenize the cleaned line into words
			for _, word := range strings.Fields(line) {
				if word != "" {

					if _, err := strconv.ParseFloat(word, 64); err == nil {
						continue
					}

					if _, exists := vocabSet[word]; !exists {
						continue
					}

					// Append to allWords slice
					allWords = append(allWords, word)
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

	return allWords, nil
}
