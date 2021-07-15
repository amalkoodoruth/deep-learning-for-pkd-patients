import numpy as np

def remove_bg_only_test(arr):

   # result = np.all((arr == 0), axis=1)
    result =  np.amax(arr) == 0
    return result

if __name__ == '__main__':
    # mlist = [0, 1, 2, 4, 6, 8, 9]
    # for i in range(11):
    #     if i in mlist:
    #         print(i, ' is in list')

    arr = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    print(remove_bg_only_test(arr))

