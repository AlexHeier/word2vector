package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

func init() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal(err.Error())
	}
}

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
	runningTime := time.Now()
	// Initialize Word2Vec model

	fmt.Print("Fetching vectors from the database\n")

	go displayTimer()
	err = w2v.OpenDB()
	if err != nil {
		fmt.Printf("Database error: %v\n", err)
		return
	}
	runTimer = false

	fmt.Printf("\nFound %d words in the database\nFetching time: %v\n", len(w2v.Vocab), time.Since(runningTime))

	var totalWords int

	runtime.GOMAXPROCS(runtime.NumCPU()) // max out all cores

	for i := 0; i < loops; i++ {
		loopTime := time.Now()
		fmt.Printf("\nStarting loop %d out of %d\n", i+1, loops)

		first := (i * 100) + 1
		last := first + 99

		if downloadNewBooks {
			deleteFolderContents(trainingdata)
			errorCount := DownloadBook(first, last, trainingdata)
			if errorCount > 0 {
				fmt.Printf("\nError downloading %d books", errorCount)
			}
		}

		fmt.Printf("\nDownloaded books %d to %d\n", first, last)

		allWords, err := w2v.preprocessText(trainingdata)
		if err != nil {
			fmt.Printf("Error preprocessing text: %v\n", err)
			return
		}

		totalWords += len(allWords) * epochs
		fmt.Printf("Found %d unique words\nFound %d words in total\nWords per epoch: %d\n", len(w2v.Vocab), totalWords, len(allWords))

		// Train the model
		w2v.TrainModel(allWords, trainingRate, epochs, threads) // Learning rate, epochs and threads

		topN := 5
		word := "fast"

		similarWords := findMostSimilarWords(word, w2v.Vectors, topN)
		fmt.Printf("Top %d words most similar to '%s': %v", topN, word, similarWords)

		result := findAnalogy("him", "man", "woman", w2v.Vectors, topN)
		fmt.Printf("\nAnalogy Test (him - man + woman): %v", result)
		fmt.Printf("\nTotal run time: %v\nEstimated time left: %v\n\n", time.Since(runningTime), time.Duration(time.Since(loopTime).Seconds()*float64(750-i)*float64(time.Second)))

		if i == 0 {
			go UpdateModelInDB()
		}
	}

	doneMain = true

	for !doneDB {
		time.Sleep(1 * time.Minute)
	}
}
