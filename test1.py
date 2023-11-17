import numpy as np

def test1func():
  print("test1func")

if __name__ == "__main__":
  print("From main test 1")
  print(np.arange(9).reshape((3,3)))
else:
  print("Not main test 1")
