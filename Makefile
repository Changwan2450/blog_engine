.DEFAULT_GOAL := help

run:
	python3 src/run_once.py --slot morning

evening:
	python3 src/run_once.py --slot evening

learn:
	@echo "Usage: make learn FILE=out/xxx.md VIEWS=1000 LIKES=20 COMMENTS=5"
	python3 src/learn.py --output_file $(FILE) --views $(VIEWS) --likes $(LIKES) --comments $(COMMENTS)

backtest:
	python3 src/backtest.py

lint:
	python3 -m py_compile src/*.py
	@echo "✅ Syntax OK"

clean:
	rm -rf __pycache__ src/__pycache__

help:
	@echo ""
	@echo "Blog Engine Commands"
	@echo ""
	@echo "make run        -> morning pipeline"
	@echo "make evening    -> evening pipeline"
	@echo "make learn      -> update bandit learning"
	@echo "make backtest   -> run strategy analysis"
	@echo "make lint       -> syntax check"
	@echo ""

.PHONY: run evening learn backtest lint clean help
