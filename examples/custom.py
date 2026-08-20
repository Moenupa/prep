from prep.api import ProcArgs, formatter


@formatter("your_dataset", "sft", "train", default_src="your/source")
def load(path: str, split: str, loadargs: ProcArgs): ...


if __name__ == "__main__":
    # redirect calls that bypass python __file__ to the CLI entrypoint
    import sys

    from typer.main import get_command

    from prep.cli import app

    # forwards arguments to the `prep` and `ppls` CLI
    # we want to keep this in one process
    # otherwise the registration will be lost
    get_command(app).main(args=sys.argv[1:], standalone_mode=False)
