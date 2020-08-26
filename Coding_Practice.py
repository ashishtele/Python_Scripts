# Two sum
def  twoSum (nums, target ):
	 d  = {}
	 for  i , n  in  enumerate ( nums ):
	     if  n  in  d : return [ d [ n ], i ]
	     d [ target - n ]=  i

twoSum([3,2,4],6)
