import argparse
import pandas as pd
import os


def sample_dataset(input_path: str, output_path: str, frac: float, seed: int):

    df = pd.read_csv(input_path)

    sampled_df = df.sample(frac=frac, random_state=seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sampled_df.to_csv(output_path, index=False)
    print(f"Збережено у: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Зменшення розміру датасету за відсотком."
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Шлях до вхідного CSV файлу"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Шлях для збереження обрізаного CSV"
    )
    parser.add_argument(
        "--frac",
        type=float,
        required=True,
        help="Частка даних для збереження (від 0.01 до 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed для відтворюваності"
    )

    args = parser.parse_args()

    if not (0.0 < args.frac <= 1.0):
        raise ValueError(
            "Параметр --frac має бути більше 0 і менше або дорівнювати 1.0 (наприклад, 0.25 для 25%)"
        )

    sample_dataset(args.input, args.output, args.frac, args.seed)
