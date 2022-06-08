
import random
import turtle

t = turtle.Turtle()
# t.pensize(5)
t.pensize(2)
t.pencolor('white')
turtle.Screen().bgcolor('black')
t.speed(15)

colors = ['red', 'blue', 'orange', 'purple']
for i in range(0, 100):
    t.pencolor(colors[random.randint(0, 3)])
    # t.circle(100)
    # t.forward(30)
    # t.left(20)
    # t.right(90)
    t.circle(100)
    t.left(10)
    t.forward(30)

turtle.done()