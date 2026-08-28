# Contributing to Prep

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help. Answering questions, helping others,
and improving the documentation are also immensely valuable.

It also helps us if you spread the word: reference the library in blog posts
about the awesome projects it made possible, or star the repository to say thank you.

**This guide was heavily inspired by [transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to Prep:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.

> All contributions are equally valuable to the community. 🥰

[prep]: https://github.com/Moenupa/prep
[prep-fork]: https://github.com/Moenupa/prep/fork

### Style guide

Prep follows the [Google Python Style Guide][google-python-styleguide] and
[PyTorch Docstring Style Guide][torch-docstring-styleguide].

[google-python-styleguide]: https://google.github.io/styleguide/pyguide.html
[torch-docstring-styleguide]: https://github.com/pytorch/pytorch/wiki/Docstring-Guidelines

### Create a Pull Request

1. Fork the [repository][prep] by clicking on the [Fork][prep-fork] button on the repository's page. This creates a copy of the code under your GitHub user account.
2. Clone your fork to your local disk, and add the base repository as a remote:
    ```bash
    git clone git@github.com:<your Github handle>/prep.git
    cd prep
    git remote add upstream https://github.com/Moenupa/prep.git
    ```

3. Create a new branch to hold your development changes:
    ```bash
    git checkout -b feat/describe_your_changes
    ```

4. Set up a development environment by running the following command in a virtual environment:
    ```bash
    uv sync --dev  # or --extra npu for NPU environment
    ```

5. Check code before commit:
    ```bash
    make format && make format-check
    make check
    make test
    ```

6. Submit changes:
    ```bash
    git add .
    git commit -m "commit message"
    git fetch upstream
    git rebase upstream/main
    git push -u origin feat/describe_your_changes
    ```

7. Create a pull request from your branch `feat/describe_your_changes` at [origin repo][prep].