#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, time, datetime, hashlib, json
import pandas as pd

def ensure_dir(path: str):
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def atomic_csv_save(df: pd.DataFrame, final_path: str, retries: int = 5, sleep_sec: float = 1.0):
    ensure_dir(final_path)
    tmp = final_path + ".tmp"
    last_err = None
    for i in range(retries):
        try:
            df.to_csv(tmp, index=False, encoding="utf-8")
            os.replace(tmp, final_path)
            return final_path
        except PermissionError as e:
            last_err = e
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            time.sleep(sleep_sec * (i + 1))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    alt = os.path.join(os.path.dirname(final_path), f"{os.path.splitext(os.path.basename(final_path))[0]}_{ts}.csv")
    df.to_csv(alt, index=False, encoding="utf-8")
    return alt

def file_fingerprint(path: str) -> str:
    st = os.stat(path)
    payload = f"{path}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def jsonl_write(records, path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

