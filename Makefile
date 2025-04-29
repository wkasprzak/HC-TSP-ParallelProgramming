TARGET = tsp

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -fopenmp -I.

SRCS = main.cpp City.cpp multiStartGreedy.cpp tabuSearch.cpp utility.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -fopenmp -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET) 10 25 30 10
