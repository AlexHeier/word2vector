package main

import (
	"database/sql"
	"sync"
)

const vectorSize int = 300 // The dementions of the vector
const windowSize int = 8   // How many words to consider left and right
const trainingdata string = "./trainingdata"
const downloadBooks bool = true
const epochs int = 10
const trainingRate float64 = 0.01
const loops int = 750

type Word2Vec struct {
	Vocab          []string
	Vectors        map[string][]float64
	UpdatedVectors map[string][]float64
	M              sync.Mutex
	DB             *sql.DB
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
