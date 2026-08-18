import pytest

from prep.api.formatutils import extract_qa


@pytest.mark.parametrize(
    ("entry", "q_cols", "a_cols", "expected_question", "expected_answer"),
    [
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "What is shown?"},
                    {"role": "assistant", "content": "A pneumothorax."},
                ]
            },
            ["question"],
            ["answer"],
            "<image><image>Q: What is shown?",
            "A: A pneumothorax.",
            id="openai-messages-format",
        ),
        pytest.param(
            {
                "conversations": [
                    {"from": "human", "value": "Summarize the finding."},
                    {"from": "gpt", "value": "Mild bibasilar atelectasis."},
                ]
            },
            ["question"],
            ["answer"],
            "<image><image>Q: Summarize the finding.",
            "A: Mild bibasilar atelectasis.",
            id="llava-conversations-format",
        ),
        pytest.param(
            {
                "prompt": [
                    {"content": "Inspect the CT."},
                    {"content": "Choose the best diagnosis."},
                ],
                "reward_model": {"ground_truth": "Pulmonary embolism"},
            },
            ["question"],
            ["answer"],
            "<image><image>Q: Inspect the CT.\n\nChoose the best diagnosis.",
            "A: Pulmonary embolism",
            id="prompt-and-reward-model-format",
        ),
        pytest.param(
            {"prompt_text": "What does the MRI show?", "label": "Normal."},
            ["question", "prompt_text"],
            ["answer", "label"],
            "<image><image>Q: What does the MRI show?",
            "A: Normal.",
            id="fallback-to-first-non-null-columns",
        ),
    ],
)
def test_extract_qa_reads_supported_input_shapes(
    entry: dict,
    q_cols: list[str],
    a_cols: list[str],
    expected_question: str,
    expected_answer: str,
) -> None:
    question, answer = extract_qa(
        entry,
        q_cols=q_cols,
        a_cols=a_cols,
        option_cols=["options", "choice_a", "choice_b"],
        q_template="Q: {question}",
        a_template="A: {answer}",
        n_img_tags=2,
    )

    assert question == expected_question
    assert answer == expected_answer


def test_extract_qa_uses_template_placeholder_without_adding_extra_image_prefix() -> (
    None
):
    question, answer = extract_qa(
        {"question": "Count the lesions.", "answer": "Two"},
        q_cols=["question"],
        a_cols=["answer"],
        option_cols=[],
        q_template="Images: {im_tags}\nQuestion: {question}",
        a_template="{answer}",
        n_img_tags=3,
    )

    assert question == "Images: <image><image><image>\nQuestion: Count the lesions."
    assert answer == "Two"


def test_extract_qa_does_not_prepend_template_image_tag_when_question_already_has_one() -> (
    None
):
    question, answer = extract_qa(
        {"question": "<image> Describe the abnormality.", "answer": "Consolidation"},
        q_cols=["question"],
        a_cols=["answer"],
        option_cols=[],
        q_template="Question: {question}",
        a_template="Answer: {answer}",
        n_img_tags=4,
    )

    assert question == "Question: <image> Describe the abnormality."
    assert answer == "Answer: Consolidation"


def test_extract_qa_converts_index_answer_to_letter_when_options_exist() -> None:
    question, answer = extract_qa(
        {
            "question": "Pick the correct diagnosis.",
            "answer": 1,
            "options": ["Edema", "Fracture", "Mass"],
        },
        q_cols=["question"],
        a_cols=["answer"],
        option_cols=["options"],
        q_template="{question}{options}",
        a_template="{answer}",
        n_img_tags=1,
    )

    assert (
        question == "<image>Pick the correct diagnosis.\nA. Edema\nB. Fracture\nC. Mass"
    )
    assert answer == "B"


@pytest.mark.parametrize(
    ("entry", "expected_error", "message_fragment"),
    [
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "One"},
                    {"role": "assistant", "content": "Two"},
                    {"role": "assistant", "content": "Three"},
                ]
            },
            NotImplementedError,
            "Expected 2 messages",
            id="rejects-messages-with-more-than-two-turns",
        ),
        pytest.param(
            {
                "conversations": [
                    {"from": "human", "value": "One"},
                ]
            },
            NotImplementedError,
            "Expected 2 conversations",
            id="rejects-conversations-with-wrong-length",
        ),
        pytest.param(
            {"question": "What is the finding?", "answer": 2.5},
            ValueError,
            "Expected answer as str",
            id="rejects-non-string-answers-without-option-index-conversion",
        ),
    ],
)
def test_extract_qa_rejects_unsupported_shapes_and_answer_types(
    entry: dict,
    expected_error: type[Exception],
    message_fragment: str,
) -> None:
    with pytest.raises(expected_error, match=message_fragment):
        extract_qa(
            entry,
            q_cols=["question"],
            a_cols=["answer"],
            option_cols=["options"],
            q_template="Q: {question}",
            a_template="A: {answer}",
            n_img_tags=1,
        )
