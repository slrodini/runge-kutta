CXX := $(shell which c++)
CXXFLAGS := -std=c++20 -Wall -Wextra -O3 -I../

EXAMPLES := $(wildcard *.cc)
BINARIES := $(EXAMPLES:.cc=)

all: $(BINARIES) compile_commands.json

%: %.cc ../runge_kutta.hh
	$(CXX) $(CXXFLAGS) -o $@ $<

compile_commands.json: $(EXAMPLES)
	@echo "[" > $@
	@dir=$$(pwd); \
	for src in $(EXAMPLES); do \
		bin=$${src%.cc}; \
		cmd="$(CXX) $(CXXFLAGS) -o $$bin $$src"; \
		printf '  { "directory": "%s", "command": "%s", "file": "%s" },\n' \
			"$$dir" "$$cmd" "$$src"; \
	done | sed '$$s/,$$//' >> $@
	@echo "]" >> $@

run-%: %
	./$<

clean:
	rm -f $(BINS)