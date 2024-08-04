import random

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
symbols = ['!', '#', '$', '%', '&', '(', ')', '*', '+']

print("Welcome to the PYPassword Generator!")
n_letters = int(input("How many letters would you like?\n"))
n_symbols = int(input("How many symbols would you like?\n"))
n_numbers = int(input("How many numbers would you like?\n"))

password = ""
for letter in range(1, n_letters + 1):
    random_letter = random.choice(letters)
    password = password + random_letter
for number in range(1, n_numbers + 1):
    random_number = random.choice(numbers)
    password = password + random_number
for symbol in range(1, n_symbols + 1):
    random_symbol = random.choice(symbols)
    password = password + random_symbol

print(f"Your password is {password}")
