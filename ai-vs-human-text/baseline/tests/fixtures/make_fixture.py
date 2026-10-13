"""Write the synthetic fixture used by the smoke test.

python tests/fixtures/make_fixture.py

The texts are generated from templates. They are not competition data, and
metrics on them say nothing about the competition. Output is deterministic.
"""

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"

HUMAN = [
    "ok so we tried the {thing} again and it kinda broke, {person} thinks its the {part}",
    "honestly the {thing} was fine until monday lol, then the {part} died",
    "anyone know why my {thing} keeps doing that? asked {person}, no idea",
    "we ran out of time so the {thing} is half done, will fix the {part} tmr",
    "lol {person} spilled coffee on the {thing}, {part} still works tho",
]
AI = [
    "The {thing} demonstrates a robust approach, and the {part} plays a pivotal role in its overall effectiveness.",
    "In conclusion, the {thing} offers significant advantages, particularly regarding the {part}, as noted by {person}.",
    "It is important to note that the {part} of the {thing} ensures a seamless and comprehensive experience.",
    "Furthermore, the {thing} leverages the {part} to deliver valuable insights, as {person} highlighted.",
    "Overall, the {thing} provides a comprehensive framework in which the {part} is essential.",
]
THINGS = ["model", "dashboard", "survey", "pipeline", "report", "prototype", "app", "experiment"]
PARTS = ["parser", "database", "login page", "chart", "scheduler", "cache", "sensor", "export"]
PEOPLE = ["Sam", "Priya", "Alex", "Wei", "Jordan", "Maria"]


def sentence(rng, templates):
    return rng.choice(templates).format(
        thing=rng.choice(THINGS), part=rng.choice(PARTS), person=rng.choice(PEOPLE)
    )


def main():
    rng = random.Random(7)
    rows = []
    for i in range(80):
        label = i % 2
        templates = AI if label else HUMAN
        text = " ".join(sentence(rng, templates) for _ in range(rng.randint(2, 4)))
        rows.append((text, label))
    rows.append((rows[0][0], rows[0][1]))  # an exact duplicate
    rows.append(("  " + rows[2][0].upper() + "  ", rows[2][1]))  # a case/space duplicate
    rows.append(("This sentence is labelled both ways.", 0))  # a conflicting pair
    rows.append(("this sentence is labelled  both ways.", 1))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label"])
        w.writerows(rows)
    with open(OUT / "test.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "text"])
        w.writerow([1, rows[5][0]])  # also in train
        for i in range(2, 11):
            w.writerow([i, " ".join(sentence(rng, AI if i % 2 else HUMAN) for _ in range(3))])


if __name__ == "__main__":
    main()
