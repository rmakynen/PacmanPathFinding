# from __future__ import print_function
import collections	#Deque. This is a higher performance version of a list. Faster when accessing first / last elements.
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Code was re-used from the util.py file 

"""
Container with LIFO policy. First element that enters will be the last one that leaves.
"""
class Stack:
    def __init__(self):
        self.deque = collections.deque()	#5,7 sec avg  of 3 runs	
	#Add new item:
    def push(self,item):
        self.deque.append(item)
	#Remove the item that was most recently pushed to the stack.
    def pop(self):
        return self.deque.pop()
	#Is the queue empty?
    def isEmpty(self):
        return len(self.deque) == 0
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Code was re-used from the util.py file 
"""
Container with FIFO policy. The first element in will also be the first element out.
"""
class Queue:
    def __init__(self):
        self.deque = collections.deque()
	#Add new item:
    def push(self,item):
        self.deque.appendleft(item)
	#Remove the oldest item in the Queue
    def pop(self):
        return self.deque.pop()
	#Is the queue empty?
    def isEmpty(self):
        return len(self.deque) == 0
# ------------------------------------------------------------------------------------------------------------------------------------------------------
"""
Container for storing X,Y and cost
"""
class Location:
    def __init__(self,x,y,param):
        self.x=x
        self.y=y
        self.param=param
        # self.deque = collections.deque()
    def getx(self):
        return self.x
    def gety(self):
        return self.y
    def getParam(self):
        return self.param
    def setx(self,x):
        self.x=x
    def sety(self,y):
        self.y=y
    def setParam(self,param):
        self.param=param
    def printLoc(self):
        print(self.getLocStr())
    def getLocStr(self):
        x = str(self.x)
        y = str(self.y)
        param = str(self.param)
        fullStr = x +","+ y +","+ param
        return fullStr
# ------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # aContainer = Queue()            #123
    aContainer = Stack()          #321
    
    for i in range(0,3):
        xCoord = 10
        yCoord = 10
        aLoc = Location(xCoord,yCoord,i+1)
        aContainer.push(aLoc)
    
    while not aContainer.isEmpty():
        item = aContainer.pop()
        item.printLoc()
if __name__ == '__main__':
    main()