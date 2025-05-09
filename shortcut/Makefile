COMPILER = g++
FLAGS = -g -O3 -march=native -std=c++17
PARALLEL_FLAG = -fopenmp
ASM_FLAG = -S
OUTPUT = shortcut.o

ifdef MP
	FLAGS += $(PARALLEL_FLAG)
endif

ifdef ASM
	FLAGS += $(ASM_FLAG)
	OUTPUT = shortcut.s
endif

baseline: baseline.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

linear: linear.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

instruction: instruction.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

vec: vec.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

register: register.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

permutation: permutation.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

prefetch: prefetch.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

cache: cache.cc
	$(COMPILER) $(FLAGS) -o $(OUTPUT) $@.cc

# may need to run with `taskset -c 0` if the CPU is heterogeneous
perf:
	perf stat -r 5 -d -e instructions -e cycles ./shortcut.o 1000

clean:
	rm -f shortcut.o shortcut.s
