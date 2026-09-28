import json
import sys
from client import Value

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "compute_autograd",
                        "description": "Evaluate y = relu(x * w + b) and compute exact gradients",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "w": {"type": "number"},
                                "b": {"type": "number"}
                            },
                            "required": ["x", "w", "b"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "compute_autograd":
            x = Value(args["x"])
            w = Value(args["w"])
            b = Value(args["b"])
            y = (x * w + b).relu()
            y.backward()
            res = {
                "output": y.data,
                "grad_x": x.grad,
                "grad_w": w.grad,
                "grad_b": b.grad
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res)}]}
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
