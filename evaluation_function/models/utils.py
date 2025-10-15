import csv
def csv_to_lists(filename: str) -> list:
    frequencies = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for key,value in reader:
            frequencies.append([key, float(value)])
    return frequencies