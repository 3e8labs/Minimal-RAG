CC = cc
CFLAGS = -O3 -march=native -ffast-math -Wall -I third_party/gte
LDFLAGS = -lm -lcurl

SOURCES = main.c chunk.c embed.c store_v1.c llm.c third_party/gte/gte.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = rag

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
