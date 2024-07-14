import nox
import os


@nox.session
def tests(session):
    # List all test files

    test_directory = "tests/handler_tests"

    test_files = [
        "arcface_handler_test.py",
        "arcface_model_test.py",
        "classifier_handler_test.py",
        "classifier_model_test.py",
        "deeplab_segmentator_handler_test.py",
        "deeplab_segmentator_model_test.py",
        "dlib_segmentator_handler_test.py",
        "dlib_segmentator_model_test.py",
        "face_detection_handler_test.py",
        "headpose_handler_test.py",
        "headpose_model_test.py",
        "ita_calculator_test.py",
        "ita_handler_test.py",
        "skincolor_handler_test.py",
        "skincolor_model_test.py",
    ]

    test_files = [os.path.join(test_directory, file) for file in test_files]

    # Run pytest for each test file
    session.run("pytest", "-vv", *test_files)


# Can take some time to run
@nox.session(name="mivolo_tests", venv_backend="venv")
def mivolo_tests(session):
    # Install pytest
    session.install("pytest")

    # Install the current project and its dependencies
    session.install("-r", "requirements.txt")

    # Install specific requirements for MiVOLO
    session.install("-r", "models/age/age_requirements.txt")

    # Run MiVOLO tests
    session.run("pytest", "-vv", "tests/handler_tests/mivolo_handler_test.py")
