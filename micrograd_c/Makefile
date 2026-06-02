CC=gcc
CFLAGS = -std=c99 -g -Wall -Wshadow --pedantic -Wvla
LIBS = -lm

OBJS = micrograd_c.o main.o
EXEC = neural_nets

all: $(OBJS)
	$(CC) $(OBJS) -o $(EXEC) $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -f *.o
	rm -f $(EXEC)