package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Print("Please provide the number of threads to use\n")
		return
	}

	threads, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Print("Please provide a valid number of threads\n")
		return
	}

	// Initialize Word2Vec model
	w2v := Word2Vec{
		Vocab:   []string{},
		Vectors: make(map[string][]float64),
		M:       sync.Mutex{},
	}

	var totalWords int

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

		w2v.LoadVectorsBinary(vectors)

		allWords, err := w2v.preprocessText(trainingdata)
		if err != nil {
			fmt.Println("Error preprocessing text:", err)
			return
		}

		// Train the model
		w2v.TrainModel(allWords, trainingRate, epochs, threads) // Learning rate, epochs and threads

		topN := 5
		word := "fast"

		totalWords += len(allWords) * epochs

		fmt.Print("Found ", len(w2v.Vocab), " unique words\nFound ", totalWords, " words in total\nWords in this iteration: ", len(allWords))

		similarWords := findMostSimilarWords(word, w2v.Vectors, topN)
		fmt.Printf("\nTop %d words most similar to '%s': %v", topN, word, similarWords)

		result := findAnalogy("him", "man", "woman", w2v.Vectors, topN)
		fmt.Printf("\nAnalogy Test (him - man + woman): %v\n\n", result)
	}
}
