# word2vector embedding

This is a training and embedding models in golang using word2vector embedding. 


## Vector data base:

### Table

This is not the most efficiant way. However, this is done due to X amount of currepted files using gob encoding from golang.

```
CREATE TABLE embeddings (
    word TEXT PRIMARY KEY,
    vector DOUBLE PRECISION[]
);

```

### .env file for PostgreSQL
```
DATABASE_PORT=
DATABASE_USER=
DATABASE_PASSWORD=
DATABASE_HOST=
DATABASE_NAME=
```

## Cross platform compile

If you want to compile the program on windows, then run it on Linux. Then you will have to set these env variables before you run the build command.
Powershell:
``` powershell
$env:GOOS="linux"
$env:GOARCH="amd64"
```