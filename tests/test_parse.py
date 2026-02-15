from musciclaim.parse import parse_model_output, try_parse_model_output
from musciclaim.schema import Decision, PromptMode


def test_parse_decision_only_ok():
    out = parse_model_output(text='{"decision":"SUPPORT"}', mode=PromptMode.D)
    assert out.decision == Decision.SUPPORT
    assert out.reasoning is None


def test_parse_reasoning_ok():
    out = parse_model_output(text='{"reasoning":"x","decision":"NEUTRAL"}', mode=PromptMode.R)
    assert out.decision == Decision.NEUTRAL
    assert out.reasoning == "x"


def test_parse_panels_ok():
    out = parse_model_output(
        text='{"figure_panels":["Panel A"],"reasoning":"x","decision":"CONTRADICT"}',
        mode=PromptMode.PANELS,
    )
    assert out.decision == Decision.CONTRADICT
    assert out.figure_panels == ["Panel A"]


def test_parse_rejects_extra_text():
    parsed, err = try_parse_model_output(text='note: {"decision":"SUPPORT"}', mode=PromptMode.D)
    assert parsed is None
    assert err


def test_parse_rejects_wrong_keys():
    parsed, err = try_parse_model_output(text='{"foo":1}', mode=PromptMode.D)
    assert parsed is None
    assert "unexpected keys" in err


def test_parse_rejects_invalid_decision():
    parsed, err = try_parse_model_output(text='{"decision":"MAYBE"}', mode=PromptMode.D)
    assert parsed is None
    assert "invalid decision" in err
