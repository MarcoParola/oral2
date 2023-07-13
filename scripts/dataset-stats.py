import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True)

args = parser.parse_args()


dataset = json.load(open(args.dataset, "r"))

category_count = dict()
for annotation in dataset["annotations"]:
    if annotation["category_id"] not in category_count:
        category_count[annotation["category_id"]] = 0
    category_count[annotation["category_id"]] += 1

for category in dataset["categories"]:
    print("-%s: %d" % (category["name"], category_count[category["id"]]))
