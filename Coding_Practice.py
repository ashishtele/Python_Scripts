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

# Sort Array By Parity

class Solution:
    def sortArrayByParity(self, A):
        beg, end = 0, len(A) - 1
        
        while beg <= end:
            if A[beg] % 2 == 0:
                beg += 1
            else:
                A[beg], A[end] = A[end], A[beg]
                end -= 1
        return A

# Binary Prefix Divisible By 5
n = [0,1,1,1,1,1]

res = []
for i in range(len(n)):
  no = int(str('0b'+''.join([ele for ele in map(str,n[:i+1])])),base=0)
  if no%5 == 0:
    res.append(True)
  else:
    res.append(False)

res

def prefixesDivBy5(self, A: List[int]) -> List[bool]:
        ans, b = [], 0
        for a in A:
            b = b << 1 | a
            ans.append(b % 5 == 0)
        return ans

# Sum of Digits in the Minimum Number
import functools
n = [34,23,1,24,75,33,54,8]

if functools.reduce(lambda x,y: x + y, list(map(int,str(min(n))))) % 2 == 0:
  print(1)
else:
  print(0)


# Decompress Run-Length Encoded List
nums = [1,2,3,4]

[x for a,b in zip(nums[0::2], nums[1::2]) for x in [b]*a]

#  Running Sum of 1d Array
def runningSum(self, A):
        return list(itertools.accumulate(A))

# Can Make Arithmetic Progression From Sequence

class Solution(object):
    def canMakeArithmeticProgression(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        arr.sort()
        k = arr[1] - arr[0]
        for i in range(1, len(arr) - 1):
            if arr[i+1] - arr[i] != k:
                return False
        return True

# Add binary
a = "11"
b = "1"

res = [0] * (max(len(a),len(b))+1)

if len(a) != len(b):
  a = a.zfill(max(len(a),len(b)))
  b = b.zfill(max(len(a),len(b)))

pos = len(res) - 1
for i in range(len(a)):

  res[pos] = res[pos] + (int(a[~i]) + int(b[~i]))
  res[pos-1] = int(res[pos]/2)
  res[pos] = res[pos]%2
  pos -= 1

res

# Repeated String Match
a = "abc"
b = "wxyz"

cnt = 0
temp = ''
thr = int((len(b)/len(a))+1)
while thr > 0:
  temp = temp + a
  cnt += 1
  if b in temp:
    print(cnt)
    break
  thr -= 1
if thr == 0:
  print(-1)

#  Unique Morse Code Words
words = ["gin", "zen", "gig", "msg"]

morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
alp = [chr(i) for i in range(97,97+26,1)]

dic = {alp:morse for alp,morse in zip(alp,morse)}
dic
ot = []
for w in words:
  res = ''
  for a in w:
    res = res + dic[a]
  ot.append(res)
len(set(ot))

def uniqueMorseRepresentations(self, words):
        d = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
             "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        return len({''.join(d[ord(i) - ord('a')] for i in w) for w in words})


# Armstrong Number

n = 123

def armstr_number(n):

  l = len(str(n))
  s = 0
  for i in str(n):
    s = s + pow(int(i), l)
  if n == s:
    print(True)
  else:
    print(False)


armstr_number(n)

# Shallow copies 
# with nested objects, modifying on level 2 or deeper does affect shallow copy

import copy
1. list_b = copy.copy(list_a)
2. list_b = list(list_a)
3. list_b = list_a[:]
4. list_b = list_a.copy()
