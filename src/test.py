def cal(depth: int = 0) -> float:
  if (depth >= 100):
    return 0

  return 2.25 + (3/5) + (2/5) * cal(depth + 1)

print(cal())