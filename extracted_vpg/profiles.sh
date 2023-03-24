python -m cProfile -o vpg_log_500.prof vpg_experiment.py
python -m cProfile -o td3_log_500.prof td3_experiment.py
git add .
git commit -m "profiling for 50 and 500 epochs"
git push

# see https://medium.com/@narenandu/profiling-and-visualization-tools-in-python-89a46f578989