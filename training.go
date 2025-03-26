package main

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// words in the list of words, learningRate is the learning rate, epochs is the number of loops to train the model
func (w2v *Word2Vec) TrainModel(words []string, learningRate float64, epochs, workers int) {
	chunkSize := len(words) / workers
	split := make([][]string, workers)

	for epoch := 0; epoch < epochs; epoch++ {
		startTime := time.Now()
		var wg sync.WaitGroup
		totalLoss := 0.0

		for w := 0; w < workers; w++ {
			start := w * chunkSize
			end := min(start+chunkSize, len(words))
			split[w] = words[start:end]

			wg.Add(1)
			go func(wordsSubset []string) {
				defer wg.Done()
				for i, target := range wordsSubset {
					start := max(0, i-windowSize)
					end := min(len(wordsSubset), i+windowSize)

					for j := start; j < end; j++ {
						if i != j {
							context := wordsSubset[j]
							w2v.M.Lock()
							loss := w2v.UpdateVectors(target, context, learningRate)
							w2v.M.Unlock()
							totalLoss += loss

						}
					}
				}
			}(split[w])
		}
		wg.Wait()

		elapsedTime := time.Since(startTime).Round(time.Millisecond).Seconds() // Sexy way to get time in seconds with only 3 decimal places
		fmt.Printf("Epoch %v out %v of took %v and had a loss of %.3f\n", epoch+1, epochs, elapsedTime, totalLoss)
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (w2v *Word2Vec) UpdateVectors(target, context string, learningRate float64) float64 {
	targetVector, targetExists := w2v.Vectors[target]
	contextVector, contextExists := w2v.Vectors[context]

	if !targetExists || !contextExists {
		return 0
	}

	// Compute dot product (similarity score)
	dotProduct := 0.0
	for i := 0; i < vectorSize; i++ {
		dotProduct += targetVector[i] * contextVector[i]
	}

	// Compute probability using sigmoid
	probability := sigmoid(dotProduct)
	bce := 1.0 - probability // binary cross-entropy

	loss := -math.Log(probability)

	// Update word vectors using gradient descent
	for i := 0; i < vectorSize; i++ {
		grad := learningRate * bce
		targetVector[i] += grad * contextVector[i]
		contextVector[i] += grad * targetVector[i]
	}

	// Save the updated vectors
	w2v.Vectors[target] = targetVector
	w2v.Vectors[context] = contextVector

	// Save the updated vectors to UpdatedVectors map
	if targetExists {
		w2v.UpdatedVectors[target] = targetVector
	}
	if contextExists {
		w2v.UpdatedVectors[context] = contextVector
	}

	return loss
}
