import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Get the params""",
    )

    parser.add_argument(
        "-J",
        "--json",
        action="store_true",
        help="Load params from json",
    )

    parser.add_argument(
        "-NL",
        "--no-log",
        action="store_true",
        help="Do not log run",
    )

    parser.add_argument(
        "-trs",
        "--train-batch-size",
        type=int,
        help="Set the train batch size",
        default=256,
        metavar="",
    )

    parser.add_argument(
        "-tes",
        "--test-batch-size",
        type=int,
        help="Set the test batch size",
        default=1024,
        metavar="",
    )

    parser.add_argument(
        "-des",
        "--dev-batch-size",
        type=int,
        help="Set the dev batch size",
        default=1024,
        metavar="",
    )

    args = vars(parser.parse_args())
    return args
