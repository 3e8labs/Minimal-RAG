# This is just for my reference - An attempt at writing proper dev log




## Version 1 — Flat Binary File

**Storage:** Custom flat binary file on disk
**Retrieval:** Brute-force linear scan with cosine similarity

### LOG

- Attempted to basically do a chunk breadkown from a file on disk.
- Did not actually do retrieval as of now.
- Just scratched the surface with producing embedings of the doc.


- Started off with --query and --k flags
- Later should implement from file?
- Decided to not make chunks out of user query.
- 

- Finished Retireval for Version 1. Which includes public api to write the store to a binary file
  Among other things. (Top_k query , write, free)
- [TO-DO] Should wire this API to main.c

