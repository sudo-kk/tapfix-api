import pytest

from app.schema import (
    Action,
    ContextDeeplinkResponse,
    Goal,
    StepGroup,
    actionCategory,
)
from app.validators import (
    scrub_urls,
    validate_description,
    validate_full_response,
    validate_goal_format,
    validate_score,
    validate_title,
)




def _make_response(
    goal: str = "Follow these steps to perform this Wi-Fi Troubleshooting",
    title: str = "Wi-Fi connectivity",
    score: float = 0.85,
    description: str = "It will resolve your connectivity issue",
    steps: list[str] | None = None,
) -> ContextDeeplinkResponse:
    """Build a minimal valid ContextDeeplinkResponse for testing."""
    if steps is None:
        steps = ["Open Settings.", "Tap Wi-Fi.", "Toggle the switch."]
    return ContextDeeplinkResponse(
        contexts=[
            Goal(
                goal=goal,
                title=title,
                score=score,
                actions=[
                    Action(
                        actionName="Fix Wi-Fi",
                        description=description,
                        stepGroups=[StepGroup(steps=steps)],
                        category=actionCategory.manual,
                    )
                ],
            )
        ]
    )




class TestValidCases:
    def test_valid_troubleshooting_goal(self):
        assert validate_goal_format(
            "Follow these steps to perform this Wi-Fi Troubleshooting"
        )

    def test_valid_configuration_goal(self):
        assert validate_goal_format(
            "Follow these steps to perform this Bluetooth Configuration"
        )

    def test_valid_full_response(self):
        resp = _make_response()
        is_valid, failures = validate_full_response(resp)
        assert is_valid, f"Expected valid, got failures: {failures}"

    def test_scrub_urls_clean_text(self):
        text = "Open Settings and tap Wi-Fi."
        assert scrub_urls(text) == text




class TestFailureCases:
    def test_bad_goal_regex(self):
        """Goal doesn't match the required sentence pattern."""
        resp = _make_response(goal="Fix the Wi-Fi please")
        is_valid, failures = validate_full_response(resp)
        assert not is_valid
        assert any("goal" in f for f in failures)
        assert not any("title" in f for f in failures)
        assert not any("score" in f for f in failures)
        assert not any("description" in f for f in failures)

    def test_four_word_title(self):
        """Title has 4 words — must be 2–3."""
        resp = _make_response(title="Wi-Fi connectivity issue fix")
        is_valid, failures = validate_full_response(resp)
        assert not is_valid
        assert any("title" in f for f in failures)
        assert not any("goal" in f for f in failures)

    def test_four_word_description(self):
        """Description has 4 words — must be 5–7 starting with 'It will'."""
        resp = _make_response(description="It will fix things")
        is_valid, failures = validate_full_response(resp)
        assert not is_valid
        assert any("description" in f for f in failures)
        assert not any("goal" in f for f in failures)

    def test_score_out_of_range(self):
        """Score of 1.5 is above the allowed 0.0–1.0 range."""
        resp = _make_response(score=1.5)
        is_valid, failures = validate_full_response(resp)
        assert not is_valid
        assert any("score" in f for f in failures)
        assert not any("goal" in f for f in failures)


class TestURLScrubbing:
    def test_step_with_embedded_url(self):
        """A step containing a URL should fail full validation."""
        resp = _make_response(
            steps=[
                "Go to https://samsung.com/support for help.",
                "Tap Wi-Fi.",
                "Toggle the switch.",
            ]
        )
        is_valid, failures = validate_full_response(resp)
        assert not is_valid
        assert any("URL" in f or "link" in f for f in failures)

    def test_scrub_https(self):
        assert scrub_urls("Visit https://samsung.com/support now") == "Visit now"

    def test_scrub_www(self):
        assert scrub_urls("Go to www.samsung.com/help page") == "Go to page"

    def test_scrub_bare_dotcom(self):
        assert scrub_urls("Check samsung.com for details") == "Check for details"

    def test_scrub_markdown_link(self):
        assert scrub_urls("Click [here](https://example.com) now") == "Click now"

    def test_scrub_markdown_image(self):
        assert scrub_urls("See ![icon](https://img.com/a.png) below") == "See below"

    def test_scrub_dothtml(self):
        assert scrub_urls("Open support/page.html for info") == "Open for info"
