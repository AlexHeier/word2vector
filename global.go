package main

import "sync"

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
