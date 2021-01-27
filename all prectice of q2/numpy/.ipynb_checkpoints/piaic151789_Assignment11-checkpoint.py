{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# AIC Q-2 Batch-8\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Name: Nazia Sajjad"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Rollno. : PIAIC151789"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment 1 For Numpy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Difficulty Level Beginner\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 1. Import the numpy package under the name np\n",
    "Ans: import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 2. Create a null vector of size 10\n",
    "Ans: a = np.zeros(10)\n",
    "print(a)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = np. zeros(10)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 3. Create a vector with values ranging from 10 to 49\n",
    "Ans: Z = np.arange(10,50)\n",
    "print(Z)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33\n",
      " 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "z = np.arange(10,50)\n",
    "print(z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 4. Find the shape of previous array in question 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(40,)\n"
     ]
    }
   ],
   "source": [
    "y = z.shape\n",
    "print(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#  Q 5. Print the type of the previous array in question 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z.dtype\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 6. Print the numpy version and the configuration\n",
    "Ans: print(np.__version__)\n",
    "np.show_config()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 7. Print the dimension of the array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "z = np.arange(10,50)\n",
    "print(z)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "z.ndim"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 8. Create a boolean array with all the True values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 1  3 11  5  9  7]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np \n",
    "a= np.array([1,3,11,5,9,7,])    \n",
    "print(a)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[11  9  7]\n"
     ]
    }
   ],
   "source": [
    "print(a[a>5])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# OR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 1]\n",
      " [4 3 6]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.array([[1, 2, 1], [4, 3, 6]])\n",
    "print(arr)\n",
    "\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 2  4  8]\n",
      " [ 7  5 12]]\n"
     ]
    }
   ],
   "source": [
    "arr2 = np.array([[2, 4, 8], [7, 5, 12]])\n",
    "print(arr2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ True  True  True]\n",
      " [ True  True  True]]\n",
      "bool\n"
     ]
    }
   ],
   "source": [
    "y= arr2 > arr\n",
    "print(y)\n",
    "print(y.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 9. Create a two dimensional array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[10 20 30]\n",
      " [40 50 60]\n",
      " [70 80 90]]\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = np.array([[10,20,30],[40,50,60],[70,80,90]])\n",
    "print(a)\n",
    "print(np.ndim(a))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 3]\n",
      " [5 7 9]]\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr = np.array([[1,2,3],[5,7,9]])\n",
    "print(arr)\n",
    "print(np.ndim(arr))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 10  20  30]\n",
      " [ 40  50  60]\n",
      " [ 70  80  90]\n",
      " [100 110 120]]\n",
      "2\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "ab = np.array([[10,20,30],[40,50,60],[70,80,90],[100,110,120]])\n",
    "print(ab)\n",
    "print(np.ndim(ab))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 10.Create a three dimensional array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[10 20 30]\n",
      "  [40 50 60]\n",
      "  [70 80 90]]]\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "x = np.array([[[10,20,30],[40,50,60],[70,80,90]]])\n",
    "print(x)\n",
    "print(np.ndim(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 1  2]\n",
      "  [ 6  7]\n",
      "  [11 12]]]\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "n = np.array([[[1,2],[6,7],[11,12]]])\n",
    "print(n)\n",
    "print(np.ndim(n))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 10  20  30  40  50]\n",
      "  [ 60  70  80  90 100]\n",
      "  [110 120 130 140 150]]]\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = np.array([[[10,20,30,40,50],[60,70,80,90,100],[110,120,130,140,150]]])\n",
    "print(a)\n",
    "print(np.ndim(a))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 10  20  30  40  50]\n",
      "  [ 60  70  80  90 100]\n",
      "  [110 120 130 140 150]\n",
      "  [160 170 180 190 200]]]\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "a = np.array([[[10,20,30,40,50],[60,70,80,90,100],[110,120,130,140,150],[160,170,180,190,200]]])\n",
    "print(a)\n",
    "print(np.ndim(a))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Difficulty Level Easy"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 11. Reverse a vector (first element becomes last)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16\n",
      " 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "Z = np.arange(40)\n",
    "Z = Z[::-1]\n",
    "print(Z)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 12. Create a null vector of size 10 but the fifth value which is 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "y = np.zeros(10)\n",
    "y[4] = 1\n",
    "print(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 13.Create a 3x3 identity matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]]\n"
     ]
    }
   ],
   "source": [
    "a = np.eye(3)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 14. Convert the data type of the given array from int to float\n",
    "arr = np.array([1, 2, 3, 4, 5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int32')"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr = np.array([1, 2, 3, 4, 5])\n",
    "arr.dtype"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 2. 3. 4. 5.]\n",
      "float64\n"
     ]
    }
   ],
   "source": [
    "arr1 = arr.astype('float64')\n",
    "print(arr1)\n",
    "print (arr1.dtype)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 15. Multiply arr1 with arr2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.  8.  3.]\n",
      " [28. 10. 72.]]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "arr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "            [4., 5., 6.]])  \n",
    "\n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "\n",
    "           [7., 2., 12.]])\n",
    "print(arr1 * arr2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 16 Make an array by comparing both the arrays provided "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 2. 3.]\n",
      " [4. 5. 6.]] [[ 0.  4.  1.]\n",
      " [ 7.  2. 12.]]\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "            [4., 5., 6.]]) \n",
    "\n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "\n",
    "            [7., 2., 12.]])\n",
    "print(arr1,arr2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False, False],\n",
       "       [False, False, False]])"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 == arr2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 2. 4.]\n"
     ]
    }
   ],
   "source": [
    "arr3=np.intersect1d(arr1,arr2)\n",
    "print(arr3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ True, False,  True],\n",
       "       [False,  True, False]])"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1>arr2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False,  True, False],\n",
       "       [ True, False,  True]])"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1<arr2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q17. Extract all odd numbers from arr with values(0-9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr = np.arange(10)\n",
    "arr\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr1 = np.where((arr >= 5))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[arr1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 18. Replace all odd numbers to -1 from previous array\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[arr %2 ==1] = -1\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 19.arr = np.arange(10)\n",
    "# Replace the values of indexes 5,6,7 and 8 to 12\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4, 12, 12, 12,  8,  9])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[5:8] = 12\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 20 Create a 2d array with 1 on the border and 0 inside\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=np.ones((10,10))\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x[1:-1,1:-1]=0 \n",
    "x"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Difficulty Level Medium\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 21 Replace the value 5 to 12"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr2d = np.array([[1, 2, 3],\n",
    "\n",
    "            [4, 5, 6], \n",
    "\n",
    "            [7, 8, 9]])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 22. Convert all the values of 1st array to 64\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  2,  3],\n",
       "        [ 4,  5,  6]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[64, 64, 64],\n",
       "        [64, 64, 64]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d[0]= 64\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6],\n",
       "       [7, 8, 9]])"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n",
    "arr2d"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 24 Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d[0, 2]\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 25 Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3, 6])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d[:2, 2]\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 26. Create a 10x10 array with random values and find the minimum and maximum values\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.007400951771163733 0.9801695592683884\n"
     ]
    }
   ],
   "source": [
    "A = np.random.random((10,10))\n",
    "Amin, Amax = A.min(), A.max()\n",
    "print(Amin, Amax)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 27 a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])¶\n",
    "Find the common items between a and b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.random.randint([1,2,3,2,3,4,3,4,5,6])\n",
    "b = np.random.randint([7,2,10,2,7,4,9,4,9,8])\n",
    "print(np.intersect1d(a,b))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 28 Find the positions where elements of a and b match"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(array([1, 3, 5, 7], dtype=int64),)\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "c= np.where(a == b)\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 29 Find all the values from array data where the values from array names are not equal to Will\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 363,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True, False,  True, False,  True,  True])"
      ]
     },
     "execution_count": 363,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "names\n",
    "names != 'Will'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.2100254 , -0.67168574,  0.34123365, -0.69498429],\n",
       "       [-0.01371164,  0.71852879,  0.58668285, -0.07138944],\n",
       "       [ 1.21989352, -0.99508901,  0.44351368,  0.5240353 ],\n",
       "       [ 1.07241421, -0.42864099, -2.17455681,  0.48889515],\n",
       "       [-0.16819669,  1.915387  ,  0.34568517,  0.53824856],\n",
       "       [-2.313477  ,  0.98167689,  0.76342881,  2.00621795],\n",
       "       [ 2.25840574, -1.251661  , -1.48028929,  0.13566236]])"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 30 Find all the values from array data where the values from array names are not equal to Will and Joe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 357,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True, False,  True, False,  True,  True])"
      ]
     },
     "execution_count": 357,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "names != 'Will'\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True, False,  True,  True,  True, False, False])"
      ]
     },
     "execution_count": 358,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names != 'Joe'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 362,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True,  True,  True,  True,  True])"
      ]
     },
     "execution_count": 362,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names != 'Will and Joe'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Difficulty Level Hard"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 31 Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4,  5,  6],\n",
       "       [ 7,  8,  9],\n",
       "       [10, 11, 12],\n",
       "       [13, 14, 15]])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "b = a.reshape((5, 3))\n",
    "b"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 32 Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "a= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  2,  3,  4],\n",
       "        [ 5,  6,  7,  8]],\n",
       "\n",
       "       [[ 9, 10, 11, 12],\n",
       "        [13, 14, 15, 16]]])"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.reshape((2,2,4))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 33 Swap axes of the array you created in Question 32\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0,  1,  2],\n",
       "       [ 3,  4,  5],\n",
       "       [ 6,  7,  8],\n",
       "       [ 9, 10, 11],\n",
       "       [12, 13, 14]])"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr = np.arange(15).reshape(5,3)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  0,  2],\n",
       "       [ 4,  3,  5],\n",
       "       [ 7,  6,  8],\n",
       "       [10,  9, 11],\n",
       "       [13, 12, 14]])"
      ]
     },
     "execution_count": 148,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr[:, [1,0,2]]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 34 Create an array of size 10, and find the square root of every element in the array,if the values less than 0.5, replace them with 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr = np.arange(10)\n",
    "arr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,\n",
       "       2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.sqrt(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "4.5"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False, False, False, False, False, False,  True,  True,  True,\n",
       "        True])"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr>5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 35 Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 354,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[  1   2   1  75  58   9   0  65  20   1  11  12]\n",
      " [  1   2  50   4   3  14   7 200   9   1  11  15]]\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.array([[1,2,1,75,58,9,0,65,20,1,11,12], [1,2,50,4,3,14,7,200,9,1,11,15]])\n",
    "print(arr1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 355,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "200"
      ]
     },
     "execution_count": 355,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1.max()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 36. Find the unique names and sort them out:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'], dtype='<U4')"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Bob', 'Joe', 'Will'], dtype='<U4')"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.unique(names)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 37 From array a remove all items present in array b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 349,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5] [5 6 7 8 9]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,4,5]) \n",
    "b = np.array([5,6,7,8,9])\n",
    "print(a,b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 351,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 2, 3, 4])"
      ]
     },
     "execution_count": 351,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.setdiff1d(a,b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 38. Following is the input NumPy array delete column two and insert following new column in its place"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[34, 43, 73],\n",
       "       [82, 22, 12],\n",
       "       [53, 94, 66]])"
      ]
     },
     "execution_count": 323,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np \n",
    "sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])\n",
    "sampleArray"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[43],\n",
       "       [22],\n",
       "       [94]])"
      ]
     },
     "execution_count": 318,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sampleArray = np.delete(sampleArray,[0,2], 1)\n",
    "sampleArray\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[10, 10, 10]])"
      ]
     },
     "execution_count": 322,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newColumn = np.array([[10,10,10]])\n",
    "newColumn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 328,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[34, 43, 73],\n",
       "       [82, 22, 12],\n",
       "       [53, 94, 66],\n",
       "       [10, 10, 10]])"
      ]
     },
     "execution_count": 328,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "newColumn = np.insert(sampleArray,3,[10],axis = 0)\n",
    "newColumn"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 39 Find the dot product of the two matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 338,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 2. 3.]\n",
      " [4. 5. 6.]] [[ 6. 23.]\n",
      " [-1.  7.]\n",
      " [ 8.  9.]]\n"
     ]
    }
   ],
   "source": [
    "x = np.array([[1., 2., 3.], [4., 5., 6.]]) \n",
    "y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "print(x,y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 340,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 28.,  64.],\n",
       "       [ 67., 181.]])"
      ]
     },
     "execution_count": 340,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.dot(x,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Q 40 Generate a matrix of 20 random values and find its cumulative sum\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "a = Generate a matrix of 20 random values and find its cumulative sum\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 346,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 1,  5,  8,  2,  9, 11, 15, 14,  2,  0,  1,  4,  5,  1,  6,  7,  2,\n",
       "        8, 10, 14])"
      ]
     },
     "execution_count": 346,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,5,8,2,9,11,15,14,2,0,1,4,5,1,6,7,2,8,10,14])\n",
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 348,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  1,   6,  14,  16,  25,  36,  51,  65,  67,  67,  68,  72,  77,\n",
       "        78,  84,  91,  93, 101, 111, 125], dtype=int32)"
      ]
     },
     "execution_count": 348,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a.cumsum()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
