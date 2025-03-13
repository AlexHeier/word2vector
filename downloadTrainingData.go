package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
)

func DownloadBook(startID, endID int, folderPath string) int {
	if startID > endID {
		temp := startID
		startID = endID
		endID = temp
	}

	errors := 0
	for bookID := startID; bookID <= endID; bookID++ {
		fmt.Printf("\rDownloading book %d/%d of total %d books", bookID, endID, endID-startID+1)

		er := false
		// Construct the URL for the book
		url := fmt.Sprintf("https://www.gutenberg.org/cache/epub/%d/pg%d.txt", bookID, bookID)

		// Make the HTTP request to download the book
		resp, err := http.Get(url)
		if err != nil {
			er = true
		}
		defer resp.Body.Close()

		// Check if the request was successful
		if resp.StatusCode != http.StatusOK {
			er = true
		}

		// Read the response body (the book content)
		content, err := io.ReadAll(resp.Body)
		if err != nil {
			er = true
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

	return errors
}
