def fibonacci(n, lookup):
	if n==0 or n==1:
		lookup[n] = n
	
	if lookup[n] is None:
		lookup[n] = fibonacci(n-1, lookup) + fibonacci(n-2, lookup)

	return lookup[n]

def main():
	n = 10
	lookup = [None]*(101)
	print('Fibonacci Number is ', fibonacci(n, lookup))

if __name__ == '__main__':
	main()