base:
	g++ -g -O3 -march=native -std=c++17 -o shortcut.o baseline.cc

asm:
	g++ -S -g -O3 -march=native -std=c++17 -o shortcut.s baseline.cc

base_mp:
	g++ -fopenmp -g -O3 -march=native -std=c++17 -o shortcut.o base_mp.cc
