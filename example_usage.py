from client import Value

def main():
    x = Value(3.0)
    w = Value(2.0)
    b = Value(1.0)
    y = (x * w + b).relu()
    y.backward()
    print("Forward y:", y.data)
    print("Gradient dy/dx (w):", x.grad)
    print("Gradient dy/dw (x):", w.grad)
    print("Gradient dy/db (1):", b.grad)

if __name__ == "__main__":
    main()
