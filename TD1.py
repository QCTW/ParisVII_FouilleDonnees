import math

from random import randint

def average(lst):
	somme = 0
	for i in lst:
		somme += i
		print("i=%d Sum=%d" %(i, somme))

	return somme / len(lst)

def median(lst):
	sorted_list = lst.sort()
	mid = int(math.floor(len(lst)/2))
	print("Mid="+str(mid)+";Len="+str(len(lst)))
	if(mid == len(lst)/2):
		return (lst[mid-1]+lst[mid])/2
	else:
		return lst[mid]

def zip (l1, l2):
	for (v1, v2) in zip(l1, l2):
		print(v1+":"+v2)

# Complexiste O(n**2)
def occur2(lst):
	return { item : lst.count(item) for item in set(lst) }

# Complexiste O(n)
def occurences(lst):
	d = {}
	for i in lst:
		if i not in d:
			d[i] = 1
		else:
			d[i] += 1		
		print(str(i)+"->"+str(d[i]))
	return d

def unique(lst):
	l = []
	dic = {}
	for i in lst:
		if i not in dic:
			dic[i] = 1
			l.append(i)
	return l

def square(lst):
	l = []
	for i in lst:
		l.append(i**2)
	return l

def stddev(lst):
	avg = average(lst)
	sizeMoins1 = len(lst)-1
	s = 0;
	for i in lst:
		s = s+(i-avg)**2

	return math.sqrt(s/sizeMoins1)
 
l = [4, 3, 2, 1, 5, 5, 5]
print("Average: "+str(average(l)))
print("Median: "+str(median(l)))

l2 = [7, 2, 3, 10, 3, 30]
print("Median: "+str(median(l2)))

l3 = [1, 1, 1, 1, 1, 1, 1, 1, 10, 10]
occurences(l2)

l4 = [20,20,20]

print(str(unique(l)))
print(str(square(l2)))
print(str(stddev(l4)))

from random import random
def uniform():
	return 0 if random()<=0.5 else 1

def uniform_test(count):
	rand_1 = 0
	for i in range(count):
		if (uniform() == 1):
			rand_1+=1
	return rand_1/float(count)

print(str(uniform()))
print(str(uniform_test(100000)))



# In theory, we will have n*p
def exam_succ(n, p):
	num_succ = 0
	for i in range(n):
		if random() > p:
			num_succ+=1
	return num_succ

def exam_succ_test():
	for i in range(100000):


# Le lio binomial
import matplotlib.pyplot as plt
plt.hist(results, 100)
plt.show()

print(str(exam_succ(1000, 0.5)))


from random import randint
def monty_hall(change):
	winning_door = randint(1,3)
	chosen_door = randint(1,3)
	return (chosen_door == winning_door and not change) or (chosen_door != winning_door and change)


def monty_hall_simulation(n, change):

