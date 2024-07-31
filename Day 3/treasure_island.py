print("Welcome to the treasure Island!")
print("Your mission is to find the treasure.")
first = input("Do you want to go left or right?\n")
if first == "left":
    print("Great\n")
    second = input("Now, Do you want to swim or wait?")
    if second == "wait":
        print("Patience is always the right way...\n")
        third = input("Now, which door do you want to enter? Red, Yellow or Blue")
        if third == "Yellow":
            print("Contratulations, you have won!")
        elif third == "Red":
            print("Burned by fire, Game over!")
        else:
            print("Game Over!")
    else:
        print("You have fall been attacked  by a trout, game over!")
else:
    print("You have fall into a hole, game over!")
