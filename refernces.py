import pandas as pd
import ast
from collections import Counter
import sys 
sys.stdout.reconfigure(encoding='utf-8')


# === Load dataset ===
csv_path = "./data/final_filtered_by_fos_and_reference.csv"
df = pd.read_csv(csv_path)

# === Filter papers published until 2009 ===
df_filtered = df[df["year"] <= 2009]

# === Collect all cited paper IDs ===
all_references = []

for ref_str in df_filtered['references']:
    try:
        refs = ast.literal_eval(ref_str)
        all_references.extend(refs)
    except:
        continue

# === Count frequency of each paper ID ===
ref_counter = Counter(all_references)
most_common_id, count = ref_counter.most_common(1)[0]

# === Display result ===
print("📌 Most Cited Paper (until 2009):")
print(f"Paper ID   : {most_common_id}")
print(f"Times Cited: {count}")
