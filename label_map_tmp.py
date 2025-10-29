import glob, pandas as pd
files = glob.glob(r"C:\Users\sujaa\PBL\DB\**\*.csv", recursive=True)
labs = set()
for f in files:
    try:
        labs.update(pd.read_csv(f, usecols=["activity"])["activity"].dropna().unique().tolist())
    except Exception:
        pass
names = sorted(labs)
print("ID -> activity")
for i, name in enumerate(names):
    print(f"{i} -> {name}")
