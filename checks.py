passed = 0
total = 0


def check(name, condition, value_text):
    # print PASS or FAIL for one check and keep a count
    global passed, total
    total += 1
    if condition:
        passed += 1
        status = "PASS"
    else:
        status = "FAIL"
    print(f"[{status}] {name}  ({value_text})")
