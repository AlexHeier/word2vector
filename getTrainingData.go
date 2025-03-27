package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

func DownloadBook(startID, endID int, folderPath, language string) int {
	if startID > endID {
		temp := startID
		startID = endID
		endID = temp
	}

	errors := 0
	differentLanguage := 0
	for bookID := startID; bookID <= endID; bookID++ {
		fmt.Printf("\rDownloading book %d/%d of total %d books", bookID, endID, endID-startID+1)

		er := false
		// Construct the URL for the book
		url := fmt.Sprintf("https://www.gutenberg.org/cache/epub/%d/pg%d.txt", bookID, bookID)

		// Make the HTTP request to download the book
		resp, err := http.Get(url)
		if err != nil {
			er = true
			continue
		}
		defer resp.Body.Close()

		// Check if the request was successful
		if resp.StatusCode != http.StatusOK {
			er = true
			continue
		}

		// Read the response body (the book content)
		content, err := io.ReadAll(resp.Body)
		if err != nil {
			er = true
			continue
		}

		if !strings.Contains(string(content), fmt.Sprintf("Language: %s", language)) {
			differentLanguage++
			continue
		}

		// Create the file to save the book
		bookFile := fmt.Sprintf("%s/%d.txt", folderPath, bookID)
		err = os.WriteFile(bookFile, content, 0644)
		if err != nil {
			er = true
		}

		if er {
			errors++
		}
	}

	fmt.Printf("\nFound %v books with language %v,\nBooks in other languages %v\n", endID-startID+1-errors-differentLanguage, language, differentLanguage)

	return errors
}

func deleteFolderContents(folderPath string) error {
	entries, err := os.ReadDir(folderPath)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		entryPath := filepath.Join(folderPath, entry.Name())
		err = os.RemoveAll(entryPath) // Removes both files and subdirectories
		if err != nil {
			fmt.Printf("Error deleting file %v\n", entryPath)
		}
	}

	return nil
}
