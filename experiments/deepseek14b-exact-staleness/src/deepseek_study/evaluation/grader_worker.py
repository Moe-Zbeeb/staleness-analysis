import json
import sys

from .grader_v3 import grade


def main():
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            result = grade(request["payload"])
            if result["status"] == "grader_error" or type(result["correct"]) is not bool:
                raise ValueError(result["reason"])
            response = {"id": request["id"], "status": "ok", "result": result}
        except Exception as error:
            response = {"id": request["id"], "status": "error", "reason": str(error)[:1000]}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
