"""Code-preference pairs: a prompt and two candidate answers, A and B.

TASKS holds every pair. NUMPY_TASKS, t11 to t15, are the round 3 pairs on numpy and pandas.
"""

TASKS = [
    {
        "id": "t01",
        "prompt": "Write a Python function that reverses a string.",
        "response_a": """```python
def reverse(s: str) -> str:
    return s[::-1]
```""",
        "response_b": """```python
def reverse(s):
    result = ""
    for ch in s:
        result = ch + result
    return result
```""",
    },
    {
        "id": "t02",
        "prompt": "Write a Python function that returns the lines of a text file as a list.",
        "response_a": """```python
def read_lines(path):
    f = open(path)
    return f.read().splitlines()
```""",
        "response_b": """```python
def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()
```""",
    },
    {
        "id": "t03",
        "prompt": "Write a Python function that fetches a user row by username from a sqlite3 connection.",
        "response_a": """```python
def get_user(conn, username):
    cur = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    )
    return cur.fetchone()
```""",
        "response_b": """```python
def get_user(conn, username):
    cur = conn.execute(
        f"SELECT * FROM users WHERE username = '{username}'"
    )
    return cur.fetchone()
```""",
    },
    {
        "id": "t04",
        "prompt": "Remove duplicates from a list while keeping the original order.",
        "response_a": """```python
def dedupe(xs):
    return list(set(xs))
```""",
        "response_b": """```python
def dedupe(xs):
    return list(dict.fromkeys(xs))
```""",
    },
    {
        "id": "t05",
        "prompt": "Write a function that appends an item to a list and returns the list. The list argument is optional.",
        "response_a": """```python
def add(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
```""",
        "response_b": """```python
def add(item, bucket=[]):
    bucket.append(item)
    return bucket
```""",
    },
    {
        "id": "t06",
        "prompt": "Wrap requests.get with retries for transient failures.",
        "response_a": """```python
import requests

def get_with_retry(url):
    while True:
        try:
            return requests.get(url)
        except Exception:
            continue
```""",
        "response_b": """```python
import time
import requests

def get_with_retry(url, attempts=3):
    for i in range(attempts):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code < 500:
                return resp
        except requests.ConnectionError:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"{url} failed after {attempts} attempts")
```""",
    },
    {
        "id": "t07",
        "prompt": "Write a function that checks whether a number is prime.",
        "response_a": """```python
def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True
```""",
        "response_b": """```python
import math

def is_prime(n: int) -> bool:
    \"\"\"Return True if n is a prime number.\"\"\"
    if n < 2:
        return False
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return False
    return True
```""",
    },
    {
        "id": "t08",
        "prompt": "Write a function that returns the average of a list of numbers. It must raise ValueError on an empty list.",
        "response_a": """```python
def average(xs):
    return sum(xs) / len(xs)
```""",
        "response_b": """```python
def average(xs):
    if not xs:
        return 0
    return sum(xs) / len(xs)
```""",
    },
    {
        "id": "t09",
        "prompt": "Give a shell one-liner that prints the total number of lines across all .py files under the current directory.",
        "response_a": """```bash
find . -name "*.py" | xargs wc -l
```""",
        "response_b": """```bash
find . -name '*.py' -print0 | xargs -0 cat | wc -l
```""",
    },
    {
        "id": "t10",
        "prompt": "In one sentence, explain what this regex matches: `^\\d{3}-\\d{4}$`",
        "response_a": "It matches a whole string made of exactly three digits, a hyphen, and exactly four digits, such as `555-1234`.",
        "response_b": """This is a regular expression. Let's break it down:

- `^` anchors the match at the start of the string.
- `\\d{3}` matches exactly three digits.
- `-` matches a literal hyphen.
- `\\d{4}` matches exactly four digits.
- `$` anchors the match at the end of the string.

So `555-1234` matches, while `5551234` and `555-12345` do not. This pattern is often used for local phone numbers.""",
    },
]

NUMPY_TASKS = [
    {
        "id": "t11",
        "prompt": "Write a pandas function that returns the DataFrame with missing values in the `country` column replaced by \"unknown\". The caller's DataFrame must not change.",
        "response_a": """```python
def fill_country(df):
    out = df
    out["country"] = out["country"].fillna("unknown")
    return out
```""",
        "response_b": """```python
def fill_country(df):
    return df.assign(country=df["country"].fillna("unknown"))
```""",
    },
    {
        "id": "t12",
        "prompt": "Write a function that returns the mean of a list of sensor readings as a float, skipping missing readings stored as NaN. If every reading is missing, return None.",
        "response_a": """```python
import numpy as np

def mean_reading(values):
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return None
    return float(np.nanmean(arr))
```""",
        "response_b": """```python
import numpy as np

def mean_reading(values):
    arr = np.asarray(values, dtype=float)
    result = np.mean(arr[arr != np.nan])
    return None if np.isnan(result) else float(result)
```""",
    },
    {
        "id": "t13",
        "prompt": "Write a pandas function that returns the average `amount` of the rows whose `status` is \"paid\". If there are no paid rows, raise ValueError.",
        "response_a": """```python
def average_paid(df):
    paid = df.loc[df["status"] == "paid", "amount"]
    if paid.empty:
        raise ValueError("no paid rows")
    return paid.mean()
```""",
        "response_b": """```python
def average_paid(df):
    paid = df[df["status"] == "paid"]
    try:
        return paid["amount"].mean()
    except ZeroDivisionError:
        raise ValueError("no paid rows")
```""",
    },
    {
        "id": "t14",
        "prompt": "Write a function that doubles the brightness of a grayscale image stored as a uint8 NumPy array. Pixels that would go above 255 must be capped at 255.",
        "response_a": """```python
import numpy as np

def brighten(img):
    return np.clip(img * 2, 0, 255).astype(np.uint8)
```""",
        "response_b": """```python
import numpy as np

def brighten(img):
    return np.clip(img.astype(np.uint16) * 2, 0, 255).astype(np.uint8)
```""",
    },
    {
        "id": "t15",
        "prompt": "Write a function that takes a 2D NumPy array of shape (n_samples, n_features) and standardises each feature (column) to mean 0 and standard deviation 1. A constant column must come out as all zeros, not NaN.",
        "response_a": """```python
import numpy as np

def standardise(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)
```""",
        "response_b": """```python
import numpy as np

def standardise(x):
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1
    return (x - mean) / std
```""",
    },
]

TASKS += NUMPY_TASKS
