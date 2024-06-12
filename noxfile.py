import nox
import os

@nox.session
def tests(session):
    session.run(
        "pytest",
        "-vv",
        "tests/handler_tests",
        "--ignore=tests/handler_tests/mivolo_handler_test.py",
    )

@nox.session
def mivolo_tests(session):
    source_env = session.posargs[0] if session.posargs else "serving"
    target_env = session.posargs[1] if len(session.posargs) > 1 else f"{source_env}_mivololo"
    
    script_path = os.path.join(os.path.dirname(__file__), "create_and_activate_conda_env.sh")

    session.run("source", script_path, source_env, target_env, external=True)
