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
    

