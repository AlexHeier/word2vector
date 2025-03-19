package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
)

func GetEnglishDictionary() error {

	url := "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt"

	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to fetch vocab: %s", resp.Status)
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		word := strings.ToLower(strings.TrimSpace(scanner.Text()))
		if word == "" {
			continue
		}
		w2v.Vectors[word] = CreateVector()
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}

func CreateVector() []float64 {
	vector := make([]float64, vectorSize)
	for i := 0; i < vectorSize; i++ {
		vector[i] = rand.Float64() * 0.1
	}
	return vector
}
