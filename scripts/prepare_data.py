"""
Script to generate sample data for VibeThinker training.
Generates data/algebra_train.jsonl and data/algebra_val.jsonl.
"""

import json
import os
import random


def generate_algebra_problem():
    """Generate a simple algebra problem."""
    a = random.randint(1, 10)
    b = random.randint(1, 20)
    c = random.randint(1, 50)
    # Equation: ax + b = c
    # Solution: x = (c - b) / a
    # Ensure integer solution
    x = random.randint(1, 10)
    c = a * x + b
    
    problem = f"Solve for x: {a}x + {b} = {c}"
    answer = str(x)
    return {"problem": problem, "answer": answer}


def main():
    """Generate train and validation datasets."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"Generating data in {data_dir}...")

    # Generate training data
    train_file = os.path.join(data_dir, "algebra_train.jsonl")
    with open(train_file, "w") as f:
        for _ in range(100):
            item = generate_algebra_problem()
            f.write(json.dumps(item) + "\n")
    print(f"✓ Generated {train_file} (100 samples)")

    # Generate validation data
    val_file = os.path.join(data_dir, "algebra_val.jsonl")
    with open(val_file, "w") as f:
        for _ in range(20):
            item = generate_algebra_problem()
            f.write(json.dumps(item) + "\n")
    print(f"✓ Generated {val_file} (20 samples)")


if __name__ == "__main__":
    main()
