# Failure scenarios and test cases

Required deliverable. Each scenario gets a test case with an expected safe
behaviour and a recorded result. Candidate scenarios come from the risks in
PROJECT_GUIDE.md. Results are filled in only from real tests.

| ID | Scenario | Expected safe behaviour | Test procedure | Result (date, pass/fail, notes) |
|---|---|---|---|---|
| F-01 | Strong window glare on the floor | No phantom/missed obstacles; slows down if confidence drops | TBD | |
| F-02 | Pane near edge-on to the camera | Still detected, or caught by the ultrasonic/e-stop | TBD | |
| F-03 | Ultrasonic dropout on an angled paper pane | Validity flag false; policy / e-stop does not trust the stale value | TBD | |
| F-04 | Pi process crash / policy hang | Pico watchdog stops the motors | TBD | |
| F-05 | Pi↔Pico link lost | Pico stops the motors | TBD | |
| F-06 | ArUco goal marker not visible | Defined behaviour (see Q12), no uncontrolled driving | TBD | |
| F-07 | Novel object (not human/car/building) | Avoided as a generic obstacle or stop | TBD | |
| F-08 | Changed daylight (different time of day) | Performance within the stated tolerance | TBD | |
