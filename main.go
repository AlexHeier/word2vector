package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const vectorSize = 300 // The dementions of the vector
const windowSize = 15  // How many words to consider left and right
const trainingdata = "./trainingdata"
const vectors = "./vectors.bin"
const downloadBooks = true

type Word2Vec struct {
	Vocab   []string
	Vectors map[string][]float64
	M       sync.Mutex
}

func (w2v *Word2Vec) InitializeVectors() {
	for _, word := range w2v.Vocab {
		vector := make([]float64, vectorSize)
		for i := 0; i < vectorSize; i++ {
			vector[i] = rand.Float64() * 0.1
		}
		w2v.Vectors[word] = vector
	}
}

func (w2v *Word2Vec) TrainModel(words []string, learningRate float64, epochs, workers int) { // words in the list of words, learningRate is the learning rate, epochs is the number of loops to train the model
	chunkSize := len(words) / workers
	split := make([][]string, workers)

	for epoch := 0; epoch < epochs; epoch++ {
		startTime := time.Now()
		var wg sync.WaitGroup

		for w := 0; w < workers; w++ {
			start := w * chunkSize
			end := min(start+chunkSize, len(words))
			split[w] = words[start:end]

			wg.Add(1)
			go func(wordsSubset []string) {
				defer wg.Done()
				for i := range wordsSubset {
					target := wordsSubset[i]
					start := max(0, i-windowSize)
					end := min(len(wordsSubset), i+windowSize)

					for j := start; j < end; j++ {
						if i != j {
							context := wordsSubset[j]
							w2v.M.Lock()
							w2v.UpdateVectors(target, context, learningRate)
							w2v.M.Unlock()

						}
					}
				}
			}(split[w])
		}
		wg.Wait()
		w2v.SaveModelBinary(vectors)

		elapsedTime := time.Since(startTime)
		fmt.Printf("\nEpoch %v out %v of took %v seconds", epoch+1, epochs, elapsedTime.Seconds())
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (w2v *Word2Vec) UpdateVectors(target, context string, learningRate float64) error {
	targetVector, targetExists := w2v.Vectors[target]
	contextVector, contextExists := w2v.Vectors[context]

	if !targetExists {
		vector := make([]float64, vectorSize)
		for i := 0; i < vectorSize; i++ {
			vector[i] = rand.Float64() * 0.1
		}
		w2v.Vectors[target] = vector
	}

	if !contextExists {
		vector := make([]float64, vectorSize)
		for i := 0; i < vectorSize; i++ {
			vector[i] = rand.Float64() * 0.1
		}
		w2v.Vectors[context] = vector
	}

	// Compute dot product (similarity score)
	dotProduct := 0.0
	for i := 0; i < vectorSize; i++ {
		dotProduct += targetVector[i] * contextVector[i]
	}

	// Compute probability using sigmoid
	probability := sigmoid(dotProduct)
	errorTerm := 1.0 - probability // Gradient of binary cross-entropy loss

	// Update word vectors using gradient descent
	for i := 0; i < vectorSize; i++ {
		grad := learningRate * errorTerm
		targetVector[i] += grad * contextVector[i]
		contextVector[i] += grad * targetVector[i]
	}

	// Save the updated vectors
	w2v.Vectors[target] = targetVector
	w2v.Vectors[context] = contextVector

	return nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

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

func main() {

	var totalWords int

	if len(os.Args) < 2 {
		fmt.Print("Please provide the number of threads to use\n")
		return
	}

	threads, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Print("Please provide a valid number of threads\n")
		return
	}

	runtime.GOMAXPROCS(runtime.NumCPU()) // max out all cores

	for i := 0; i < 750; i++ {

		deleteFolderContents(trainingdata)

		first := (i * 100) + 1
		last := first + 99

		if downloadBooks {
			errorCount := DownloadBook(first, last, trainingdata)
			if errorCount > 0 {
				fmt.Printf("\nError downloading %d books", errorCount)
			}
		}

		fmt.Printf("\nDownloaded books %d to %d\n", first, last)

		// Initialize Word2Vec model
		w2v := Word2Vec{
			Vocab:   []string{},
			Vectors: make(map[string][]float64),
			M:       sync.Mutex{},
		}

		w2v.LoadVectorsBinary(vectors)

		allWords, err := w2v.preprocessText(trainingdata)
		if err != nil {
			fmt.Println("Error preprocessing text:", err)
			return
		}

		// Initialize vectors
		w2v.InitializeVectors()

		// Train the model
		w2v.TrainModel(allWords, 0.1, 10, threads) // Learning rate, epochs and threads

		topN := 5
		word := "fast"

		totalWords += len(allWords)

		fmt.Print("Found ", len(w2v.Vocab), " unique words\nFound ", totalWords, " words in total\nWords in this iteration: ", len(allWords))

		similarWords := findMostSimilarWords(word, w2v.Vectors, topN)

		fmt.Printf("\nTop %d words most similar to '%s':\n", topN, word)
		for _, w := range similarWords {
			fmt.Println(w)
		}

		result := findAnalogy("him", "man", "woman", w2v.Vectors, topN)
		fmt.Println("Analogy Test (him - man + woman):", result)
	}
}

func findMostSimilarWords(word string, wordVecs map[string][]float64, topN int) []string {
	var similarities []struct {
		word       string
		similarity float64
	}

	// Loop through the word vectors and calculate cosine similarity
	for w, vec := range wordVecs {
		if w != word {
			similarity := cosineSimilarity(wordVecs[word], vec)
			similarities = append(similarities, struct {
				word       string
				similarity float64
			}{w, similarity})
		}
	}

	// Sort by similarity (descending)
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].similarity > similarities[j].similarity
	})

	// Get top N most similar words
	var topWords []string
	for i := 0; i < topN && i < len(similarities); i++ {
		topWords = append(topWords, similarities[i].word)
	}

	return topWords
}

func cosineSimilarity(vec1, vec2 []float64) float64 {
	var dotProduct, magnitude1, magnitude2 float64

	for i := range vec1 {
		dotProduct += vec1[i] * vec2[i]
		magnitude1 += vec1[i] * vec1[i]
		magnitude2 += vec2[i] * vec2[i]
	}

	if magnitude1 == 0 || magnitude2 == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(magnitude1) * math.Sqrt(magnitude2))
}

func findAnalogy(wordA, wordB, wordC string, wordVecs map[string][]float64, topN int) []string {
	vecA, okA := wordVecs[wordA]
	vecB, okB := wordVecs[wordB]
	vecC, okC := wordVecs[wordC]
	if !okA || !okB || !okC {
		return nil
	}

	// Compute analogy vector
	analogyVec := make([]float64, len(vecA))
	for i := range vecA {
		analogyVec[i] = vecA[i] - vecB[i] + vecC[i]
	}

	// Find most similar words
	return findMostSimilarWordsFromVector(analogyVec, wordVecs, topN)
}

// Find closest words to a vector
func findMostSimilarWordsFromVector(targetVec []float64, wordVecs map[string][]float64, topN int) []string {
	var similarities []struct {
		word       string
		similarity float64
	}

	for word, vec := range wordVecs {
		sim := cosineSimilarity(targetVec, vec)
		similarities = append(similarities, struct {
			word       string
			similarity float64
		}{word, sim})
	}

	// Sort by similarity
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].similarity > similarities[j].similarity
	})

	// Get top N words
	var topWords []string
	for i := 0; i < topN && i < len(similarities); i++ {
		topWords = append(topWords, similarities[i].word)
	}

	return topWords
}

func deleteFolderContents(folderPath string) error {
	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		entryPath := filepath.Join(folderPath, entry.Name())
		err = os.RemoveAll(entryPath) // Removes both files and subdirectories
		if err != nil {
			return err
		}
	}

	return nil
}
