package main

import "sync"

const vectorSize = 300 // The dementions of the vector
const windowSize = 8   // How many words to consider left and right
const trainingdata = "./trainingdata"
const vectors = "./vectors.bin"
const downloadBooks = true
const epochs = 10
const trainingRate = 0.01

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
