# warm up isaac sim
python tests/benchmark/profiling.py -s Rs_int
rm output.json
# 1st batch: baselines
python tests/benchmark/profiling.py -f                  # baseline (fastest config possible)
python tests/benchmark/profiling.py -s Rs_int -f        # for vision research
python tests/benchmark/profiling.py -s Rs_int -r 1      # for robotics research
python tests/benchmark/profiling.py -s Rs_int -r 3      # for multi-agent research

# 2nd batch: compare different scenes
python tests/benchmark/profiling.py -r 1 -s house_single_floor -f
python tests/benchmark/profiling.py -r 1 -s grocery_store_cafe -f
python tests/benchmark/profiling.py -r 1 -s Pomaria_0_garden -f

# 3rd batch: OG non-physics features
python tests/benchmark/profiling.py -r 1 -w             # fluids (water)
python tests/benchmark/profiling.py -r 1 -c             # soft body (cloth)
python tests/benchmark/profiling.py -r 1 -p             # macro particle system (diced objects)
python tests/benchmark/profiling.py -r 1 -w -c -p       # everything