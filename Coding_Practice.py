# Two sum
def  twoSum (nums, target ):
	 d  = {}
	 for  i , n  in  enumerate ( nums ):
	     if  n  in  d : return [ d [ n ], i ]
	     d [ target - n ]=  i

twoSum([3,2,4],6)

# Binary Flip
def bin_flip(n):

  return bin(255 - int(n,2))[2:]

bin_flip('01010101')

# longest common prefix
class Solution:
    def longestCommonPrefix(self, m):
        if not m: return ''
				#since list of string will be sorted and retrieved min max by alphebetic order
        s1 = min(m)
        s2 = max(m)

        for i, c in enumerate(s1):
            if c != s2[i]:
                return s1[:i] #stop until hit the split index
        return s1

# no of bits

  def hammingWeight(self, n):
        result = 0
        while n:
            n &= n - 1
            result += 1
        return result

# Count Largest Group
import functools
import collections
from collections import defaultdict
# method 1:
dict = {ele : [] for ele in range(1,11)}
# method 2:
dict = defaultdict(list)

for i in range(1,2+1):
  k = functools.reduce(lambda x, y: x+y,list(map(int,list(str(i)))))

  #dict[k].append(i)
  dict[k].append(i)

len([ele for ele in dict.values() if len(ele) == len(max([ele for ele in dict.values()], key = len))])

# Check If a String Can Break Another String
s1 = 'abc'
s2 =  'xya'
all([x <= y for x,y in zip(*sorted([sorted(s1),sorted(s2)]))])

# pascal triangle

# pascal = [[1]*(i+1) for i in range(numRows)]
n = 5
res = [[1],[1,1]]
for i in range(2,n):
  temp = []
  temp.append(1)
  for j in range(1,i):
    print(i,j)
    temp.append(res[i-1][j-1] + res[i-1][j])
  temp.append(1)
  res.append(temp)
res

row = [1]
n = 3
for _ in range(n):
  row = [x + y for x,y in zip([0]+row, row+[0])]
row

# Best Time to Buy and Sell Stock
class Solution(object):
    def maxProfit(self, prices):
        max_profit, min_price = 0, float("inf")
        for price in prices:
            min_price = min(min_price, price)
            max_profit = max(max_profit, price - min_price)
        return max_profit

# Shortest Unsorted Continuous Subarray
nums = [2, 6, 4, 8, 10, 15]

res = [i for (i, (a, b)) in enumerate(zip(nums, sorted(nums)))  if a != b]
[0 if not res else res[-1] - res[0] + 1]
