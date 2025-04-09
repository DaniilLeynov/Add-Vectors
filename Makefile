CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lOpenCL
TARGET = addvectors
SOURCES = addvectors.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
