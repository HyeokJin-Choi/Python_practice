# Python_practice
파이썬 공부장

https://drive.google.com/drive/search?q=owner:me%20(type:application/vnd.google.colaboratory%20||%20type:application/vnd.google.colab)

insert(i,x) --> x를 리스트의 i번 원소로 삽입한다.
append(x) --> 원소 x를 리스트의 맨 뒤에 추가한다.
pop(i) --> 리스트의 i번 원소를 삭제하면서 알려준다.
remove(x) --> 리스트에서 (처음으로 나타나는) x를 삭제한다.
index(x) --> 원소 x가 리스트의 몇 번 원소인지 알려준다.
clear() --> 리스트를 깨끗이 청소한다.
count(x) --> 리스트에서 원소 x가 몇 번 나타나는지 알려준다.
extend(a) --> 리스트에 나열할 수 있는 객체 a를 풀어서 추가한다
copy() --> 리스트를 복사한다.
reverse() --> 리스트의 순서를 역으로 뒤집는다.
sort() --> 리스트의 원소들을 정렬한다.

데이터구조 단방향리스트

from list listNode in=mport ListNode
from typing import Tuple

class LinkedListBasic:
  def __init__(self):
    self.__head = ListNode('Dummy',None)
    self.__numItems = 0
    
  def insert(self,i:int,newItem):
      if(i>=0 and i<=self.__numItems):
        prev = self.__getNode(i-1)
        newNode = ListNode(newItem,prev.next)
        prev.next = newNode
        self.__numItems += 1
       else:
        print("index",i,"값이 범위를 벗어 났습니다.")
  def append(self,newItem):

def mergeSort(A, p:int, r:int):   
    if p < r:
        q = (p+r)//2 
        mergeSort(A, p, q)
        mergeSort(A, q+1, r)
        merge(A,p,q,r)
        
def merge(A, p:int, r:int):  #(A,0,len(A)-1)이 들어갈 값
    tmp = []
    q = (p+r)//2
    while p < q and q < r: 
        if A[p] < A[p+q]:
            tmp[p] = A[p]
            p += 1; q +=1
        else:
            tmp[p] = A[p+q]
            p += 1; q += 1
위 코드의 문제점
merge 함수의 파라미터 개수가 맞지 않습니다. merge(A, p, r) 대신 merge(A, p, q, r)로 수정해야 합니다.

merge 함수에서 tmp 리스트를 초기화하지 않았습니다. 따라서 tmp 리스트에 값을 추가하려고 할 때 IndexError가 발생할 것입니다. tmp = []와 같이 리스트를 초기화해야 합니다.

merge 함수에서 p와 q는 왼쪽 리스트의 인덱스이지만, q와 r은 오른쪽 리스트의 인덱스입니다. 따라서 tmp 리스트에 값을 추가할 때 p+q 대신 q+1을 사용해야 합니다.

merge 함수에서 비교 연산자가 잘못 사용되었습니다. if A[p] < A[p+q]: 대신 if A[p] < A[q+1]:으로 수정해야 합니다.

import random
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def selectionSortRec(A, n):         # 길이 n을 가지는 A[0,1,..n-1] 리스트의 선택정렬(재귀)
    if (n>1):
        k= theLargest4Rec(A, n)   # 0에서 n까지에서의 최대값 찾기
        A[k], A[n-1] = A[n-1], A[k] # 최대값 원소 A[k]와 마지막 원소 A[n-1] 교환
        selectionSortRec(A, n-1)    # 원소의 길이가 하나 줄어든 자신의 재귀호출

def theLargest4Rec(A, last:int) -> int:	# A[0...last]에서 가장 큰 수의 인덱스를 리턴한다
	largest = 0
	for i in range(last):          # A[0,..last] 까지의 최대값 찾기
		if A[i] > A[largest]:
			largest = i
	return largest

def bubbleSortRec(A, n):    # A[0,..,n-1] 리스트의 버블정렬(재귀)
    for i in range(n-1):    # n-1번 반복
        if A[i] > A[i+1]:   # 현재 요소가 다음 요소보다 크면
            A[i], A[i+1] = A[i+1], A[i]   # 두 요소를 교환

    if n > 1:               # 리스트 크기가 2보다 큰 경우에만
        bubbleSortRec(A, n-1)   # 재귀적으로 호출하여 다음 요소들을 정렬

def insertionSortRec(A, n):
    if n <= 1:  # 종료 조건: 원소의 개수가 1 이하면 함수 종료
        return

    insertionSortRec(A, n - 1)  # 재귀 호출: 마지막 원소를 제외한 나머지 원소들을 정렬

    key = A[n - 1]  # 삽입할 원소를 key 변수로 저장
    j = n - 2  # key 원소의 왼쪽 인덱스로 초기화

    while j >= 0 and A[j] > key:  # key 원소를 삽입할 위치를 찾기 위해 왼쪽으로 이동하며 비교
        A[j + 1] = A[j]  # key보다 큰 원소들을 오른쪽으로 한 칸씩 이동
        j -= 1  # 인덱스를 왼쪽으로 이동하여 비교 대상을 한 칸씩 왼쪽으로 이동

    A[j + 1] = key  # key 원소를 적절한 위치에 삽입

def mergeSort(A, p:int, r:int):   
    if p < r: #A배열의 원소가 1개 이상일 시 진행
        q = (p+r)//2 #A배열의 중간지점
        mergeSort(A, p, q) #A배열의 왼쪽 절반
        mergeSort(A, q+1, r) #A배열의 오른쪽 절반
        merge(A,p,q,r)
        
def merge(A, p:int, q:int, r:int):  #배열, 시작점, 중간점, 끝점
    tmp = [] #배열하나 생성(정렬)
    i = p #시작점
    j = q+1 #중간점+1
    while i <= q and j <= r: #시작점이 중간점까지, 중간점은 끝지점까지 갈 때까지 반복
        if A[i] < A[j]: #시작점의 원소값이 중간점의 원소값보다 작을 시
            tmp.append(A[i]) #tmp배열에 i번째 원소값을 append
            i += 1 #i+1
        else: #시작점의 원소값이 중간점의 원소값보다 크거나 같을 시
            tmp.append(A[j]) #tmp배열에 j번째(중간점)의 원소값을 append
            j += 1 #j+1
    while i <= q: #정렬 후 왼쪽 반에 남은 원소
        tmp.append(A[i]) #tmp배열 마지막에 추가
        i += 1 
    while j <= r: #정렬 후 오른쪽 반에 남은 원소
        tmp.append(A[j]) #tmp배열 마지막에 추가
        j += 1
    for i in range(p, r+1): # =len(A)만큼 반복
        A[i] = tmp[i-p] #정렬된 tmp배열의 원소들을 A배열에 순서대로 넣음

def quickSort(A, p:int, r:int):
    # 범위 [p, r]이 한 개 이상의 원소를 갖는 경우에만 수행
    if p<r:
        # 배열 A를 분할하는 pivot 위치 q를 계산
        q=quickSortPartition(A,p,r)
        # 분할된 두 영역에 대해 quickSort 함수를 재귀적으로 호출
        quickSort(A,p,q-1) # 첫번째 영역
        quickSort(A,q+1,r) # 두번째 영역

# 배열 A를 분할하는 함수
def quickSortPartition(A, p:int, r:int):
    # 기준 원소로 A[r]을 선택
    x=A[r] 
    # 분할 위치를 기억하는 변수 i를 초기화
    i=p-1
    # A[p]~A[r-1]까지의 모든 원소에 대해 반복문을 수행
    for j in range(p,r):
        # 현재 원소 A[j]가 기준 원소 A[r]보다 작은 경우
        if x>A[j]: 
            # 분할 위치 i를 1 증가시키고, A[i]와 A[j]를 교환
            i+=1 
            A[i], A[j]=A[j], A[i] 
    # 분할 위치 바로 다음 위치와 A[r]을 교환하여 분할을 완료
    A[i+1], A[r]=A[r], A[i+1] 
    # 분할 위치를 반환
    return i+1

def heapify(A, n, i):
    largest = i  # 최대값을 현재 노드로 설정
    left = 2 * i + 1  # 왼쪽 자식 노드 인덱스
    right = 2 * i + 2  # 오른쪽 자식 노드 인덱스

    # 왼쪽 자식 노드가 힙의 범위 내에 있고 현재 노드보다 크면 최대값을 갱신
    if left < n and A[left] > A[largest]:
        largest = left

    # 오른쪽 자식 노드가 힙의 범위 내에 있고 현재 노드보다 크면 최대값을 갱신
    if right < n and A[right] > A[largest]:
        largest = right

    # 최대값이 현재 노드가 아니라면 노드 교환 및 재귀적으로 heapify 호출
    if largest != i:
        A[i], A[largest] = A[largest], A[i]
        heapify(A, n, largest)

def heapSort(A):
    n = len(A)

    # 최대 힙을 구성
    for i in range(n // 2 - 1, -1, -1):
        heapify(A, n, i)

    # 힙 정렬 수행
    for i in range(n - 1, 0, -1):
        A[0], A[i] = A[i], A[0]  # 최대값을 배열의 마지막 위치로 이동
        heapify(A, i, 0)  # 힙 크기를 줄이고 재구성

def shellSort(A):
    N = len(A)
    h = N // 2  # 초기 간격 설정

    while h > 0:
        for i in range(h, N):
            temp = A[i]  # 현재 정렬 대상을 임시 변수에 저장
            j = i - h  # 이전 위치에 대한 인덱스 설정

            while j >= 0 and A[j] > temp:
                A[j + h] = A[j]  # 이전 위치에 있는 값을 현재 위치로 이동
                j -= h  # 간격만큼 이전으로 이동

            A[j + h] = temp  # 삽입 대상을 적절한 위치에 삽입
        h //= 2  # 간격을 줄여가며 반복

sys.setrecursionlimit(1000000)
listLength = 300

B = []
for value in range(0, listLength):
    B.append(random.randint(0, 100))

Mtimes = []
for _ in range(1000):
    arr = B.copy()
    start_time = time.time()
    mergeSort(arr, 0, len(arr) - 1)
    end_time = time.time()
    Mtimes.append(end_time - start_time)
Stimes =[]
for _ in range(1000):
    arr = B.copy() 
    start_time = time.time()
    selectionSortRec(arr,len(arr))
    end_time = time.time()
    Stimes.append(end_time - start_time)  
Btimes =[]
for _ in range(1000):
    arr = B.copy() 
    start_time = time.time()
    bubbleSortRec(arr, len(arr))
    end_time = time.time()
    Btimes.append(end_time - start_time)
Itimes=[]
for _ in range(1000):
    arr = B.copy() 
    start_time = time.time()
    insertionSortRec(arr, len(arr))
    end_time = time.time()
    Itimes.append(end_time - start_time)
Qtimes=[]
for _ in range(1000):
    arr = B.copy() 
    start_time = time.time()
    quickSort(arr, 0, len(arr)-1)
    end_time = time.time()
    Qtimes.append(end_time - start_time)   
Htimes=[]
for _ in range(1000):
    arr = B.copy() 
    start_time = time.time()
    heapSort(arr)
    end_time = time.time()
    Htimes.append(end_time - start_time)
SHtimes=[]
for _ in range(1000):
    arr = B.copy() 
    start_time = time.time()
    shellSort(arr)
    end_time = time.time()
    SHtimes.append(end_time - start_time)

df = pd.DataFrame()
df['bubbleSort'] = pd.DataFrame(Btimes)
df['selectionSort'] = pd.DataFrame(Stimes)
df['mergeSort'] = pd.DataFrame(Mtimes)
df['insertionSort'] = pd.DataFrame(Itimes)
df['quickSort'] = pd.DataFrame(Qtimes)
df['heapSort'] = pd.DataFrame(Htimes)
df['shellSort'] = pd.DataFrame(SHtimes)
df.mean().plot(kind='bar')

plt.show()
