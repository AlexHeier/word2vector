package main

import (
	"database/sql"
	"fmt"
	"sync"
	"time"
)

const vectorSize int = 300 // The dementions of the vector
const windowSize int = 8   // How many words to consider left and right
const trainingdata string = "./trainingdata"
const downloadBooks bool = true
const epochs int = 10
const trainingRate float64 = 0.01
const loops int = 750

var runTimer bool = true

var dbM sync.Mutex
var DBCon *sql.DB

type Word2Vec struct {
	Vocab          []string
	Vectors        map[string][]float64
	UpdatedVectors map[string][]float64
	M              sync.Mutex
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

func displayTimer() {
	startTime := time.Now()
	for runTimer {
		fmt.Printf("\rTime Elapsed: %v\n", time.Since(startTime))
		time.Sleep(1 * time.Second)
	}
	runTimer = true
}
