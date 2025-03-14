package main

import (
	"math"
	"sort"
)

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
