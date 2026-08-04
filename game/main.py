import turtle
import numpy as np
import random



# Create screen
screen = turtle.Screen()
screen.title("My First Turtle Program")
screen.bgcolor("white")
print("Width :", screen.window_width())
print("Height:", screen.window_height())

mass1 = 10
mass2 = 20

# Create turtle
m1 = turtle.Turtle()
m1.shape("circle")
m1.penup()
m1.color("blue")
m1.pensize(2)
m1.speed(0)
m1.goto(random.randint(-screen.window_width()//2, screen.window_width()//2), random.randint(-screen.window_height()//2, screen.window_height()//2))


m2 = turtle.Turtle()
m2.shape("circle")
m2.penup()  
m2.color("red")
m2.pensize()
m2.speed(0)
m2.goto(random.randint(-screen.window_width()//2, screen.window_width()//2), random.randint(-screen.window_height()//2, screen.window_height()//2))

distance = m1.distance(m2)  # Calculate the distance between m1 and m2



# # Draw square
# for i in range(4):
#     m1.forward(100)
#     m1.right(90)

# Keep window open
turtle.done()



