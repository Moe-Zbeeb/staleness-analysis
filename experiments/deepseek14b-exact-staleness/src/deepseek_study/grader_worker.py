import contextlib
import json
import sys

from math_verify.errors import TimeoutException

from deepseek_study.rewards import ReferenceRejected, grade_result, normalize_reference, parse_gold


def main():
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if request["operation"] == "reference":
                    parse_gold(request["answer"], request["timeout"])
                    result = {"normalized_reference": normalize_reference(request["answer"])}
                elif request["operation"] == "grade":
                    result = grade_result(**request["arguments"])
                else:
                    raise ValueError("Unknown grader operation")
            response = {"status": "ok", "result": result}
        except ReferenceRejected:
            response = {"status": "reference_rejected", "reason": "unsupported_reference"}
        except (Exception, TimeoutException) as error:
            response = {"status": "error", "reason": type(error).__name__}
        print(json.dumps({"id": request["id"], **response}), flush=True)


if __name__ == "__main__":
    main()
