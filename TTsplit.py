import os
import shutil
import random

classes=["up","down","left","right"]

# ----------- CLEAR OLD DATA -----------
shutil.rmtree("data/train",ignore_errors=True)
shutil.rmtree("data/val",ignore_errors=True)

for c in classes:
    os.makedirs(f"data/train/{c}",exist_ok=True)
    os.makedirs(f"data/val/{c}",exist_ok=True)
# -------------------------------------


for c in classes:

    src=f"data/raw/{c}"

    files=[
    f for f in os.listdir(src)
    if f.endswith(".jpg")
    ]

    random.shuffle(files)

    split=int(0.8*len(files))

    train=files[:split]
    val=files[split:]

    for f in train:
        shutil.copy(
        f"{src}/{f}",
        f"data/train/{c}/{f}"
        )

    for f in val:
        shutil.copy(
        f"{src}/{f}",
        f"data/val/{c}/{f}"
        )

print("Split Complete")