import csv

def print_first_n_rows_csv(filepath, n=5):
    """
    Prints the first 'n' rows of a CSV file.

    Args:
        filepath (str): The path to the CSV file.
        n (int): The number of rows to print (default is 5).
    """
    try:
        with open(filepath, mode='r', newline='') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i < n:
                    print(row)
                else:
                    break
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

print("Printing first 5 rows of 'ocean_features_projected.csv':")
print_first_n_rows_csv("ocean_features_projected.csv", n=5)

