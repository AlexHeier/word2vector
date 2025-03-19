# Word2Vector Embedding

This project is a word to vector embedding training in Golang using the skip-gram model.

## Usage

### Supporting Other Languages

By default, the training is set up for English. To change the language, modify the following:

- Replace `GetEnglishDictionary()` with a function that retrieves a dictionary for the target language.
- Modify the `DownloadBook()` function to download books in the desired language.

Currently, `DownloadBook()` uses Project Gutenberg, a public domain library. Ensure that the chosen language has sufficient resources available in this library.

### Dependencies
This project relies on the following external dependencies:

- [`github.com/joho/godotenv`](https://github.com/joho/godotenv) – For loading environment variables from a `.env` file.
- [`github.com/lib/pq`](https://github.com/lib/pq) – PostgreSQL driver for Golang.

To install these dependencies, run:
```sh
go get github.com/joho/godotenv github.com/lib/pq
```

### PostgreSQL Table Structure

The program stores word vectors in a PostgreSQL database due to size limitations in Golang's `gob` binary encoding.

#### Table Schema:
```sql
CREATE TABLE embeddings (
    word TEXT PRIMARY KEY,
    vector DOUBLE PRECISION[]
);

CREATE UNIQUE INDEX embeddings_word_idx ON embeddings(word);
```

### Configuring PostgreSQL with `.env`
Create a `.env` file to store database credentials:
```env
DATABASE_PORT=
DATABASE_USER=
DATABASE_PASSWORD=
DATABASE_HOST=
DATABASE_NAME=
```

## Cross-Platform Compilation

To compile the program on Windows and run it on Linux, set the following environment variables before building:

### PowerShell:
```powershell
$env:GOOS="linux"
$env:GOARCH="amd64"
```

Then, run the build command as usual.

