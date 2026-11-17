# Makefile
# type 'make install' then 'make run'
# (no make on Windows? run: pip install -r requirements.txt  then  python scripts/run_eda.py)

install:
	pip install -r requirements.txt

run:
	python scripts/run_eda.py

clean:
	rm -f results/*.csv
	rm -f figures/*.png
	rm -f folds/*.json
	rm -f data/processed/*.csv
