package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"
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

	runningTime := time.Now()

	var totalWords int

	runtime.GOMAXPROCS(runtime.NumCPU()) // max out all cores

	for i := 0; i < 750; i++ {
		loopTime := time.Now()

		deleteFolderContents(trainingdata)

		first := (i * 100) + 1
		last := first + 1

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
			fmt.Printf("Error preprocessing text: %v\n", err)
			return
		}

		totalWords += len(allWords) * epochs
		fmt.Printf("Found %d unique words\nFound %d words in total\nWords in this iteration: %d\n", len(w2v.Vocab), totalWords, len(allWords))

		// Train the model
		w2v.TrainModel(allWords, trainingRate, epochs, threads) // Learning rate, epochs and threads

		topN := 5
		word := "fast"

		similarWords := findMostSimilarWords(word, w2v.Vectors, topN)
		fmt.Printf("Top %d words most similar to '%s': %v", topN, word, similarWords)

		result := findAnalogy("him", "man", "woman", w2v.Vectors, topN)
		fmt.Printf("\nAnalogy Test (him - man + woman): %v", result)
		fmt.Printf("\nTotal run time: %v\nEstimated time left: %v\n\n", time.Since(runningTime), time.Duration(time.Since(loopTime).Seconds()*float64(750-i)*float64(time.Second)))
	}
}
