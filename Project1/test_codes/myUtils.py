import main
# ------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Function for testing performance of Queue and Stack
Using Deque instead of list

Call by: 
aContainer = Queue()
performance_testing(aContainer)

OR:
aContainer = Stack()
performance_testing(aContainer)

'''
def performance_testing(aContainer):
	import time
	start_time = time.time()
	times = 10000000
	for i in range(times):
		aContainer.push(i)	#append to bottom
	for i in range(times):
		a=aContainer.pop()	# print("Thingy: "+str(a))
	time_spent = time.time() - start_time
	print("Time spent: "+ str(time_spent))
# ------------------------------------------------------------------------------------------------------------------------------------------------------