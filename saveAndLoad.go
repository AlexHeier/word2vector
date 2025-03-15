package main

import (
	"database/sql"
	"fmt"
	"os"
	"strings"
	"time"

	_ "github.com/lib/pq"
)

// Save word embeddings using HDF5
func (w2v *Word2Vec) SaveVectors() error {
	// Convert all words and vectors into slices
	var values []interface{}
	var placeholders []string
	i := 1

	// Iterate over all words and vectors in the map
	for word, vector := range w2v.UpdatedVectors {
		// Convert vector to PostgreSQL array format
		vecStr := "{" + strings.Trim(strings.Replace(fmt.Sprint(vector), " ", ",", -1), "[]") + "}"

		// Add the values to the slices for batch insert
		placeholders = append(placeholders, fmt.Sprintf("($%d, $%d)", i, i+1))
		values = append(values, word, vecStr)
		i += 2
	}

	// Function to insert in batches
	insertBatch := func(values []interface{}) error {
		// Construct the SQL query for batch insert
		query := fmt.Sprintf(`
			INSERT INTO embeddings (word, vector) 
			VALUES %s
			ON CONFLICT (word) DO UPDATE 
			SET vector = EXCLUDED.vector
		`, strings.Join(placeholders[:len(values)/2], ", "))

		// Execute the batch insert
		_, err := w2v.DB.Exec(query, values...)
		return err
	}

	// Determine the batch size based on PostgreSQL's max parameter limit
	batchSize := 65535 / 2 // Each word and vector requires 2 parameters (word + vector)

	// Insert in smaller batches
	for start := 0; start < len(values); start += batchSize * 2 {
		end := start + batchSize*2
		if end > len(values) {
			end = len(values)
		}

		// Slice the values for this batch and insert
		batchValues := values[start:end]
		if err := insertBatch(batchValues); err != nil {
			return fmt.Errorf("failed to save vectors in batch: %v", err)
		}
	}

	return nil
}

// OpenDB establishes a connection to the database, retrieves all vectors, and populates the struct
func (w2v *Word2Vec) OpenDB() error {
	// Load connection details from environment variables
	host := os.Getenv("DATABASE_HOST")
	port := os.Getenv("DATABASE_PORT")
	user := os.Getenv("DATABASE_USER")
	password := os.Getenv("DATABASE_PASSWORD")
	dbname := os.Getenv("DATABASE_NAME")
	sslmode := "disable"

	// Create the connection string
	connStr := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s", host, port, user, password, dbname, sslmode)

	// Open connection to the database
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return fmt.Errorf("failed to connect to database: %v", err)
	}

	// Store the connection in the Word2Vec object
	w2v.DB = db

	// Load vectors from the database
	err = w2v.loadVectors()
	if err != nil {
		return fmt.Errorf("failed to load vectors: %v", err)
	}

	return nil
}

// loadVectors retrieves all vectors and words from the database and populates w2v.Vectors and w2v.Vocab
func (w2v *Word2Vec) loadVectors() error {
	// Query to fetch all words and their corresponding vectors
	rows, err := w2v.DB.Query("SELECT word, vector FROM embeddings")
	if err != nil {
		return fmt.Errorf("failed to query vectors: %v", err)
	}
	defer rows.Close()

	// Initialize the Vectors map and Vocab slice
	w2v.Vectors = make(map[string][]float64)
	w2v.Vocab = []string{}

	// Loop over the query results and populate the map and slice
	for rows.Next() {
		var word string
		var vectorStr string

		// Scan the word and vector from the database
		err := rows.Scan(&word, &vectorStr)
		if err != nil {
			return fmt.Errorf("failed to scan row: %v", err)
		}

		// Convert the PostgreSQL array string to a slice of floats
		vector := parseVector(vectorStr)

		// Add the word to the vocabulary and the vector to the Vectors map
		w2v.Vectors[word] = vector
		w2v.Vocab = append(w2v.Vocab, word)
	}

	// Check if there was an error during row iteration
	if err := rows.Err(); err != nil {
		return fmt.Errorf("error iterating over rows: %v", err)
	}

	return nil
}

// parseVector converts a PostgreSQL array string to a slice of floats
func parseVector(vectorStr string) []float64 {
	// Remove the curly braces and split by commas
	vectorStr = strings.Trim(vectorStr, "{}")
	vectorParts := strings.Split(vectorStr, ",")

	// Convert each part to a float64
	var vector []float64
	for _, part := range vectorParts {
		var value float64
		fmt.Sscanf(part, "%f", &value)
		vector = append(vector, value)
	}

	return vector
}

func (w2v *Word2Vec) UpdateModelInDB() {
	startTime := time.Now()
	fmt.Print("Saving vectors to the database\n")
	err := w2v.SaveVectors()
	if err != nil {
		fmt.Printf("failed to save vectors: %v\n", err)
	}
	fmt.Printf("Saving vectors took %v seconds\n", time.Since(startTime).Seconds())

	w2v.UpdatedVectors = make(map[string][]float64)
}
