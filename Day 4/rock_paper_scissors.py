import random

rock = '''
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
'''

paper = '''
    _______
---'   ____)____
          ______)
          _______)
         _______)
---.__________)
'''

scissors = '''
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
'''


computer = random.randint(0,2)
choice = int(input("What do you want to choose? Type 0 for rock, 1 for paper and 2 for scissors\n"))
if choice == 0 and computer == 0:
    print("You chose Rock")
    print(rock)
    print("Computer chose Rock!")
    print(rock)
    print("\nIt's a tie!")
elif choice == 0 and computer == 1:
    print("You chose Rock")
    print(rock)
    print("Computer chose Paper!")
    print(paper)
    print("\nYou Loose!")
elif choice == 0 and computer == 2:
    print("You chose Rock")
    print(rock)
    print("Computer chose Scissors!")
    print(scissors)
    print("\nYou win!")
elif choice == 1 and computer == 0:
    print("You chose Paper")
    print(paper)
    print("Computer chose Rock!")
    print(rock)
    print("\nYou win!")
elif choice == 1 and computer == 1:
    print("You chose Paper!")
    print(paper)
    print("Computer chose Paper!")
    print(paper)
    print("\nIt's a tie!")
elif choice == 1 and computer == 2:
    print("You chose Paper")
    print(paper)
    print("Computer chose Scissors!")
    print(scissors)
    print("\nYou Loose!")
elif choice == 2 and computer == 0:
    print("You chose Scissors")
    print(scissors)
    print("Computer chose Rock!")
    print(rock)
    print("\nYou Loose!")
elif choice == 2 and computer == 1:
    print("You chose Scissors")
    print(scissors)
    print("Computer chose Paper!")
    print(paper)
    print("\nYou Win!")
else:
    print("You chose Scissors")
    print(scissors)
    print("Computer chose Scissors!")
    print(scissors)
    print("\nIt's a Tie!")
    
    