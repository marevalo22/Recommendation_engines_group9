# %% [markdown] cell 1
# # football_recsys_draft_augmented
# 
#     **Group 9 contributors**
# 
# - ANDREA BASTIEN SAXOD
# - CLOÉ KARINA CHAPOTOT
# - CONSTANTIN NICOLA HATECKE
# - MARCELA MENDES GIMENES FUNABASHI
# - MATIAS VICENTE AREVALO MARTINEZ
# - VITTORIO FIALDINI
# 
# 
#     This overview notebook is a compact, submission-safe replacement for the earlier augmented draft notebook. It gives a quick cross-track view of the preserved original benchmark, the reproduced real-only benchmark, and the synthetic proof-of-concept outputs.

# %% cell 2
import warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
from IPython.display import display, Markdown

NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = NOTEBOOK_DIR if (NOTEBOOK_DIR / 'data' / 'real').exists() else NOTEBOOK_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from submission_helpers import load_report_ready_tables, load_synthetic_outputs
syn = load_synthetic_outputs(PROJECT_ROOT)
tables = load_report_ready_tables(PROJECT_ROOT)

# %% [markdown] cell 3
# ## Preserved original draft benchmark

# %% cell 4
display(tables['baseline'])

# %% [markdown] cell 5
# ## Reproduced real-only benchmark and synthetic proof-of-concept comparison

# %% cell 6
display(tables['real_only'])
display(tables['synthetic_base'])
display(tables['synthetic_poc'])

# %% [markdown] cell 7
# ## Delta vs real-only benchmark

# %% cell 8
display(tables['delta'])

# %% [markdown] cell 9
# ## Named-club demos

# %% cell 10
display(syn['demo_arsenal'])
display(syn['demo_girona'])

