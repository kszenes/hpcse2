BINARY = messages
CXXFLAGS = -std=c++11 -Wno-narrowing -Wall -Wextra
CXXFLAGS += -O3

.PHONY: $(BINARY)

$(BINARY): messages.cpp
	mpicxx -o $@ $< $(CXXFLAGS)

# # $(BINARY): messages.cpp
# # 	mpicxx -o $@ $< 
run:
	mpirun -n 2 $(BINARY)

.PHONY: clean
clean:
	rm $(BINARY)
