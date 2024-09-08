import concurrent.futures
def f(x):
        return x * x
def main():

    nums = [1,2,3,4,5,6,7,8,9,10]


    # Make sure the map and function are working
    print([val for val in map(f, nums)])

    # Test to make sure concurrent map is working
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for item in executor.map(f, nums):
            print(item)

if __name__ == '__main__':
    main()