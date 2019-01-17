#performance considerations

deque used instead of list
Performance increase of a factor of 58 (0.057 sec vs 3,24 sec) for push and pop operations for the "Stack" Class. Performance improvement for the "Stack" class was ~14% (when compared to a normal list).







import sys
#Loop inputs
# for i,anItem in enumerate(sys.argv):
	# print("arg["+str(i)+"] == \""+str(anItem)+str("\""))
	
import collections	#Deque. This is a higher performance version of a list

import time


# de = collections.deque()
de = []

start_time = time.time()

times = 60000
for i in range(times):
	#list 
	de.append(i)	#append to bottom
	# de[:0] = [i]

# for i in range(times):
	# de.peek()
	#de[i]
	# de.popleft()

time_spent = time.time() - start_time
print("Time spent: "+ str(time_spent))

#When appending several elements and then popping all.
# 31.85  = dequeue is faster than list. 
# 34.652 = list









#test cases for the Location class:
# aLoc = Location(10,10,"afds")
# a = aLoc.getx()
# print("X-Value: "+str(aLoc.getx()))
# print("Y-Value: "+str(aLoc.gety()))
# print("param:   "+str(aLoc.getParam()))
# print(aLoc.getLocStr())
# aLoc.printLoc()